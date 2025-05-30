# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, NamedTuple, Set
import asyncio
import json
import re
import uvicorn
import logging
from collections import defaultdict
import bittensor as bt
import threading
from datetime import datetime
import time
from contextlib import asynccontextmanager
import psutil
from concurrent.futures import ThreadPoolExecutor
from config import config
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caches
process_cache = {"data": [], "last_updated": None, "is_updating": False, "error": None}
log_cache = {}
metagraph_cache = {}
previous_processes = {}

# Locks
cache_lock = threading.Lock()
log_cache_lock = threading.Lock()
metagraph_cache_lock = threading.Lock()

# Thread pool for subprocess commands
pm2_executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_SUBPROCESSES)

# Global background task references
background_task = None
metagraph_task = None


# Optimized data structures
class CompactProcessInfo(NamedTuple):
    """Memory-efficient process info"""

    id: int
    state: str
    args: List[str]
    error_logs: str = ""
    out_logs: str = ""

    def to_dict(self):
        return {
            "Id": self.id,
            "State": self.state,
            "Args": self.args,
            "Error logs": self.error_logs,
            "Out logs": self.out_logs,
        }


class CompactResponseData(NamedTuple):
    """Memory-efficient response data"""

    id: int
    name: str
    state: str
    uid: int
    netuid: str
    wallet_name: str
    hotkey: str
    hotkey_ss58: str
    miner_uid: str
    port: str
    stake: float
    emission: float
    error_logs: str = ""
    out_logs: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "uid": self.uid,
            "netuid": self.netuid,
            "wallet_name": self.wallet_name,
            "hotkey": self.hotkey,
            "hotkey_ss58": self.hotkey_ss58,
            "miner_uid": self.miner_uid,
            "port": self.port,
            "stake": self.stake,
            "emission": self.emission,
            "daily_alpha": self.emission * 20,
            "error_logs": self.error_logs,
            "out_logs": self.out_logs,
        }


class AdaptiveSyncScheduler:
    """Adjusts sync frequency based on activity"""

    def __init__(self):
        self.base_interval = config.PROCESS_SYNC_INTERVAL
        self.current_interval = config.PROCESS_SYNC_INTERVAL
        self.last_request_time = time.time()
        self.request_count = 0

    def get_next_interval(self):
        """Increase interval if no requests, decrease if active"""
        if not config.ADAPTIVE_SYNC:
            return self.base_interval

        time_since_request = time.time() - self.last_request_time

        if time_since_request > 300:  # No requests for 5 minutes
            self.current_interval = min(180, int(self.current_interval * 1.2))
        elif time_since_request < 30:  # Recent activity
            self.current_interval = max(30, int(self.current_interval * 0.9))
        else:
            # Gradually return to base interval
            self.current_interval = int(
                self.current_interval * 0.95 + self.base_interval * 0.05
            )

        return self.current_interval

    def mark_request(self):
        self.last_request_time = time.time()
        self.request_count += 1


sync_scheduler = AdaptiveSyncScheduler()


class MetagraphCache:
    """Caches metagraph data with independent sync schedule"""

    def __init__(self):
        self.cache = {}
        self.last_sync = {}
        self.lock = threading.Lock()
        self.sync_in_progress = {}

    def is_sync_needed(self, netuid: str) -> bool:
        """Check if sync is needed (used by process sync)"""
        with self.lock:
            if netuid not in self.last_sync:
                return True
            # Only sync if data is stale
            return time.time() - self.last_sync[netuid] > config.METAGRAPH_CACHE_TTL

    def force_sync(self, netuid: str):
        """Force a sync by clearing last sync time"""
        with self.lock:
            if netuid in self.last_sync:
                del self.last_sync[netuid]

    async def get_metagraph_data_async(self, netuid: int, force: bool = False):
        """Get cached or fresh metagraph data asynchronously"""
        netuid_str = str(netuid)

        # Check if sync is already in progress for this netuid
        with self.lock:
            if self.sync_in_progress.get(netuid_str, False):
                logger.info(f"Sync already in progress for netuid {netuid}, waiting...")
                # Wait a bit and return cached data
                await asyncio.sleep(1)
                return self.cache.get(netuid_str)

        if force or self.is_sync_needed(netuid_str):
            try:
                with self.lock:
                    self.sync_in_progress[netuid_str] = True

                logger.info(f"Starting metagraph sync for netuid {netuid}")

                # Run sync in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                start_time = time.time()

                metagraph = await loop.run_in_executor(
                    None, lambda: bt.metagraph(netuid=netuid)
                )
                await loop.run_in_executor(None, metagraph.sync)

                sync_duration = time.time() - start_time

                with self.lock:
                    self.cache[netuid_str] = {
                        "hotkeys": metagraph.hotkeys,
                        "stakes": metagraph.S,
                        "emissions": metagraph.E,
                        "sync_duration": sync_duration,
                    }
                    self.last_sync[netuid_str] = time.time()
                    self.sync_in_progress[netuid_str] = False

                logger.info(
                    f"Metagraph synced for netuid {netuid} in {sync_duration:.2f}s"
                )

            except Exception as e:
                logger.error(f"Failed to sync netuid {netuid}: {e}")
                with self.lock:
                    self.sync_in_progress[netuid_str] = False
                return self.cache.get(netuid_str)

        with self.lock:
            return self.cache.get(netuid_str)


metagraph_cache_instance = MetagraphCache()


async def metagraph_sync_loop():
    """Dedicated background loop for metagraph syncing every 5 minutes"""
    # Wait 30 seconds before first sync to let processes initialize
    await asyncio.sleep(30)

    while True:
        try:
            logger.info("=== Starting scheduled metagraph sync (5 minute interval) ===")
            sync_start_time = time.time()

            # Get all unique netuids from current processes
            active_netuids = set()

            with cache_lock:
                current_data = process_cache.get("data", [])

            for process in current_data:
                netuid = process.get("netuid")
                if netuid and process.get("state") == "online":
                    try:
                        active_netuids.add(int(netuid))
                    except (ValueError, TypeError):
                        pass

            if active_netuids:
                logger.info(
                    f"Found {len(active_netuids)} active netuids to sync: {sorted(active_netuids)}"
                )

                # Force sync all active netuids
                tasks = []
                for netuid in sorted(active_netuids):
                    # Force sync by passing force=True
                    task = metagraph_cache_instance.get_metagraph_data_async(
                        netuid, force=True
                    )
                    tasks.append((netuid, task))

                # Wait for all syncs to complete
                if tasks:
                    results = await asyncio.gather(
                        *[task for _, task in tasks], return_exceptions=True
                    )

                    success_count = 0
                    failed_netuids = []

                    for (netuid, _), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            logger.error(f"Failed to sync netuid {netuid}: {result}")
                            failed_netuids.append(netuid)
                        elif result is not None:
                            success_count += 1

                    sync_duration = time.time() - sync_start_time
                    logger.info(
                        f"=== Metagraph sync completed in {sync_duration:.2f}s: "
                        f"{success_count}/{len(tasks)} successful ==="
                    )

                    if failed_netuids:
                        logger.warning(f"Failed netuids: {failed_netuids}")
            else:
                logger.info("=== No active netuids to sync ===")

        except Exception as e:
            logger.error(f"Error in metagraph sync loop: {e}")

        # Calculate time to next sync to ensure exactly 5 minute intervals
        elapsed = time.time() - sync_start_time
        sleep_time = max(0, config.METAGRAPH_SYNC_INTERVAL - elapsed)

        logger.info(f"Next metagraph sync in {sleep_time:.1f} seconds")
        await asyncio.sleep(sleep_time)


def extract_arg_value(args: List[str], arg_name: str) -> Optional[str]:
    """Extract value for a specific argument from args list"""
    try:
        idx = args.index(arg_name)
        if idx + 1 < len(args):
            return args[idx + 1]
    except ValueError:
        pass
    return None


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text"""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


async def run_subprocess_async(cmd: List[str], timeout: int = None) -> tuple:
    """Run subprocess command asynchronously"""
    if timeout is None:
        timeout = config.SUBPROCESS_TIMEOUT

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        return stdout.decode(), stderr.decode(), proc.returncode
    except asyncio.TimeoutError:
        if proc:
            proc.kill()
            await proc.wait()
        raise TimeoutError(f"Command {' '.join(cmd)} timed out")


async def get_pm2_processes_async() -> Dict[str, CompactProcessInfo]:
    """Get all PM2 processes asynchronously"""
    try:
        if config.USE_ASYNC_SUBPROCESS:
            stdout, stderr, returncode = await run_subprocess_async(["pm2", "jlist"])
            if returncode != 0:
                raise Exception(f"PM2 command failed: {stderr}")
            processes_data = json.loads(stdout)
        else:
            # Fallback to sync version in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                pm2_executor,
                lambda: subprocess.run(
                    ["pm2", "jlist"], capture_output=True, text=True, check=True
                ),
            )
            processes_data = json.loads(result.stdout)

        process_info = {}

        for process in processes_data:
            id = process.get("pm_id", 0)
            name = process.get("name", "")
            status = process.get("pm2_env", {}).get("status", "unknown")
            args = process.get("pm2_env", {}).get("args", [])

            process_info[name] = CompactProcessInfo(id=id, state=status, args=args)

        return process_info

    except Exception as e:
        logger.error(f"Failed to get PM2 processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_cached_logs(name: str, log_type: str = "error") -> str:
    """Get logs with caching"""
    cache_key = f"{name}_{log_type}"
    current_time = time.time()

    with log_cache_lock:
        if cache_key in log_cache:
            cached_data = log_cache[cache_key]
            if current_time - cached_data["timestamp"] < config.LOG_CACHE_TTL:
                return cached_data["logs"]

    # Fetch new logs
    try:
        cmd = [
            "pm2",
            "logs",
            name,
            "--lines",
            str(config.MAX_LOG_LINES),
            "--raw",
            "--nostream",
        ]
        if log_type == "error":
            cmd.append("--err")

        if config.USE_ASYNC_SUBPROCESS:
            stdout, stderr, _ = await run_subprocess_async(cmd)
            output = stdout + stderr
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                pm2_executor,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.SUBPROCESS_TIMEOUT,
                ),
            )
            output = result.stdout + result.stderr

        cleaned_output = strip_ansi(output)

        # Update cache
        with log_cache_lock:
            log_cache[cache_key] = {"logs": cleaned_output, "timestamp": current_time}

        return cleaned_output

    except Exception as e:
        logger.warning(f"Failed to get {log_type} logs for {name}: {e}")
        return f"Error fetching logs: {str(e)}"


def get_process_changes(old_processes: dict, new_processes: dict) -> dict:
    """Detect which processes have changed"""
    changes = {"added": [], "removed": [], "state_changed": [], "all_changed": False}

    if not old_processes:
        changes["all_changed"] = True
        changes["added"] = list(new_processes.keys())
        return changes

    old_names = set(old_processes.keys())
    new_names = set(new_processes.keys())

    changes["added"] = list(new_names - old_names)
    changes["removed"] = list(old_names - new_names)

    for name in old_names & new_names:
        if old_processes[name].state != new_processes[name].state:
            changes["state_changed"].append(name)

    return changes


async def sync_processes_async():
    """Optimized async process synchronization"""
    global process_cache, previous_processes

    logger.info("Starting async process synchronization...")

    try:
        with cache_lock:
            process_cache["is_updating"] = True
            process_cache["error"] = None

        # Get current processes
        processes = await get_pm2_processes_async()

        # Detect changes if incremental updates are enabled
        changes = None
        if config.ENABLE_INCREMENTAL_UPDATES and previous_processes:
            changes = get_process_changes(previous_processes, processes)
            if not changes["all_changed"] and not any(
                [changes["added"], changes["removed"], changes["state_changed"]]
            ):
                logger.info("No process changes detected, skipping full update")
                with cache_lock:
                    process_cache["is_updating"] = False
                return

        # Prepare data structures
        netuid_wallets = defaultdict(set)
        active_netuids = set()

        # First pass: identify active netuids
        for name, info in processes.items():
            if info.state == "online":
                netuid = extract_arg_value(info.args, "--netuid")
                wallet_name = extract_arg_value(info.args, "--wallet.name")

                if netuid and wallet_name:
                    netuid_wallets[netuid].add(wallet_name)
                    active_netuids.add(int(netuid))

        # Fetch metagraph data from cache (don't force sync here)
        metagraph_results = {}
        for netuid in active_netuids:
            # Get cached data without forcing sync
            cached_data = await metagraph_cache_instance.get_metagraph_data_async(
                netuid, force=False
            )
            if cached_data:
                metagraph_results[str(netuid)] = cached_data

        # Process logs - only for changed processes or all if first run
        log_tasks = []
        processes_to_log = set()

        if changes and not changes["all_changed"]:
            processes_to_log = set(changes["added"] + changes["state_changed"])
        else:
            processes_to_log = set(processes.keys())

        # Create log fetching tasks
        for name in processes_to_log:
            if name in processes:
                error_task = get_cached_logs(name, "error")
                out_task = get_cached_logs(name, "out")
                log_tasks.append((name, "error", error_task))
                log_tasks.append((name, "out", out_task))

        # Gather logs concurrently
        log_results = {}
        if log_tasks:
            results = await asyncio.gather(
                *[task for _, _, task in log_tasks], return_exceptions=True
            )
            for (name, log_type, _), result in zip(log_tasks, results):
                if not isinstance(result, Exception):
                    key = f"{name}_{log_type}"
                    log_results[key] = result

        # Build response data
        responses: List[CompactResponseData] = []

        for name, info in processes.items():
            try:
                # Get logs from results or cache
                error_logs = log_results.get(f"{name}_error", "")
                out_logs = log_results.get(f"{name}_out", "")

                # If not in results, try to get from previous data
                if not error_logs and name in previous_processes:
                    for cached_resp in process_cache.get("data", []):
                        if cached_resp.get("name") == name:
                            error_logs = cached_resp.get("error_logs", "")
                            out_logs = cached_resp.get("out_logs", "")
                            break

                # Extract process arguments
                netuid = extract_arg_value(info.args, "--netuid")
                wallet_name = extract_arg_value(info.args, "--wallet.name")
                hotkey = extract_arg_value(info.args, "--wallet.hotkey")
                miner_uid = extract_arg_value(info.args, "--miner_uid")
                port = extract_arg_value(info.args, "--axon.port")

                uid = -1
                stake = 0.0
                emission = 0.0
                hotkey_ss58 = ""

                if netuid and wallet_name and hotkey:
                    try:
                        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
                        hotkey_ss58 = wallet.hotkey.ss58_address

                        # Get metagraph data
                        metagraph_data = metagraph_results.get(netuid)
                        if metagraph_data and info.state == "online":
                            try:
                                # Use actual wallet address
                                uid = metagraph_data["hotkeys"].index(hotkey_ss58)
                                stake = float(metagraph_data["stakes"][uid])
                                emission = float(metagraph_data["emissions"][uid])
                            except (ValueError, IndexError):
                                logger.debug(
                                    f"Hotkey not found in metagraph for {name}"
                                )
                    except Exception as e:
                        logger.warning(f"Error processing wallet for {name}: {e}")

                response_data = CompactResponseData(
                    id=info.id,
                    name=name,
                    state=info.state,
                    uid=uid,
                    netuid=netuid or "",
                    wallet_name=wallet_name or "",
                    hotkey=hotkey or "",
                    hotkey_ss58=hotkey_ss58,
                    miner_uid=miner_uid or "",
                    port=port or "",
                    stake=stake,
                    emission=emission,
                    error_logs=error_logs,
                    out_logs=out_logs,
                )
                responses.append(response_data)

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                # Add minimal data on error
                responses.append(
                    CompactResponseData(
                        id=info.id,
                        name=name,
                        state=info.state,
                        uid=-1,
                        netuid="",
                        wallet_name="",
                        hotkey="",
                        hotkey_ss58="",
                        miner_uid="",
                        port="",
                        stake=0.0,
                        emission=0.0,
                        error_logs=f"Processing error: {str(e)}",
                        out_logs="",
                    )
                )

        # Update cache
        with cache_lock:
            process_cache["data"] = [resp.to_dict() for resp in responses]
            process_cache["last_updated"] = datetime.now()
            process_cache["is_updating"] = False
            process_cache["error"] = None

        # Store current processes for next comparison
        previous_processes = processes

        logger.info(f"Process sync completed. Found {len(responses)} processes.")

    except Exception as e:
        logger.error(f"Error during process synchronization: {e}")
        with cache_lock:
            process_cache["is_updating"] = False
            process_cache["error"] = str(e)


async def background_sync_loop_async():
    """Async background sync loop with adaptive scheduling"""
    while True:
        try:
            await sync_processes_async()

            # Get next interval from adaptive scheduler
            next_interval = sync_scheduler.get_next_interval()
            logger.info(f"Next process sync in {next_interval} seconds")

            await asyncio.sleep(next_interval)

        except Exception as e:
            logger.error(f"Unexpected error in process sync loop: {e}")
            await asyncio.sleep(30)  # Wait before retry


def get_resource_usage():
    """Monitor application resource usage"""
    try:
        process = psutil.Process()
        return {
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "threads": process.num_threads(),
            "open_files": (
                len(process.open_files()) if hasattr(process, "open_files") else 0
            ),
        }
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global background_task, metagraph_task

    # Startup
    logger.info("Starting up PM2 Process Monitor...")

    # Run initial sync
    await sync_processes_async()

    # Start background process sync loop
    background_task = asyncio.create_task(background_sync_loop_async())
    logger.info("Background process synchronization started")

    # Start metagraph sync loop - runs every 5 minutes strictly
    metagraph_task = asyncio.create_task(metagraph_sync_loop())
    logger.info("Metagraph sync scheduled every 5 minutes (strict interval)")

    # Log initial resource usage
    resources = get_resource_usage()
    logger.info(f"Initial resource usage: {resources}")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Cancel background tasks
    tasks_to_cancel = []

    if background_task:
        tasks_to_cancel.append(background_task)

    if metagraph_task:
        tasks_to_cancel.append(metagraph_task)

    for task in tasks_to_cancel:
        task.cancel()

    # Wait for all tasks to complete cancellation
    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    # Close thread pool
    pm2_executor.shutdown(wait=True)

    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(title="PM2 Process Monitor", version="2.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_processes():
    """Main endpoint to get all PM2 process information"""
    logger.info("HTTP request received")

    # Mark request for adaptive scheduling
    sync_scheduler.mark_request()

    with cache_lock:
        cache_data = process_cache.copy()

    # Return cached data with metadata
    response = {
        "data": cache_data["data"],
        "last_updated": (
            cache_data["last_updated"].isoformat()
            if cache_data["last_updated"]
            else None
        ),
        "is_updating": cache_data["is_updating"],
        "error": cache_data["error"],
        "stats": {
            "process_count": len(cache_data["data"]),
            "sync_interval": sync_scheduler.current_interval,
            "resources": get_resource_usage(),
        },
    }

    # If no data yet and not updating, return informative message
    if not cache_data["data"] and not cache_data["is_updating"]:
        logger.info("No cached data available, triggering sync...")
        # Trigger async sync without waiting
        asyncio.create_task(sync_processes_async())

        return {
            **response,
            "message": "Initial synchronization in progress, please retry in a few seconds",
        }

    return response


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    resources = get_resource_usage()

    with cache_lock:
        last_updated = process_cache.get("last_updated")
        is_updating = process_cache.get("is_updating", False)
        has_error = bool(process_cache.get("error"))

    # Check metagraph sync health
    metagraph_health = {}
    with metagraph_cache_instance.lock:
        current_time = time.time()
        for netuid, last_sync_time in metagraph_cache_instance.last_sync.items():
            time_since_sync = current_time - last_sync_time
            # Warn if sync is overdue by more than 1 minute
            is_healthy = time_since_sync < (config.METAGRAPH_SYNC_INTERVAL + 60)
            metagraph_health[netuid] = {
                "healthy": is_healthy,
                "minutes_since_sync": round(time_since_sync / 60, 1),
                "overdue": not is_healthy,
            }

    # Check if last update was too long ago
    unhealthy_threshold = 300  # 5 minutes
    time_since_update = None
    is_healthy = True

    if last_updated:
        time_since_update = (datetime.now() - last_updated).total_seconds()
        if time_since_update > unhealthy_threshold:
            is_healthy = False

    # Check if any metagraph is unhealthy
    metagraph_is_healthy = all(m["healthy"] for m in metagraph_health.values())

    return {
        "status": (
            "healthy"
            if is_healthy and not has_error and metagraph_is_healthy
            else "unhealthy"
        ),
        "last_update_seconds_ago": time_since_update,
        "is_updating": is_updating,
        "has_error": has_error,
        "resources": resources,
        "metagraph_health": metagraph_health,
        "config": {
            "process_sync_interval": sync_scheduler.current_interval,
            "metagraph_sync_interval": config.METAGRAPH_SYNC_INTERVAL,
            "max_log_lines": config.MAX_LOG_LINES,
            "async_enabled": config.USE_ASYNC_SUBPROCESS,
        },
    }


@app.get("/metagraph-status")
async def get_metagraph_status():
    """Get detailed metagraph cache status"""
    with metagraph_cache_instance.lock:
        cache_status = {}
        current_time = time.time()

        for netuid, last_sync_time in metagraph_cache_instance.last_sync.items():
            time_since_sync = current_time - last_sync_time
            next_sync_in = max(0, config.METAGRAPH_SYNC_INTERVAL - time_since_sync)

            cache_data = metagraph_cache_instance.cache.get(netuid, {})

            cache_status[netuid] = {
                "last_sync": datetime.fromtimestamp(last_sync_time).isoformat(),
                "seconds_since_sync": round(time_since_sync),
                "minutes_since_sync": round(time_since_sync / 60, 1),
                "next_sync_in_seconds": round(next_sync_in),
                "next_sync_in_minutes": round(next_sync_in / 60, 1),
                "cached_data_available": bool(cache_data),
                "sync_in_progress": metagraph_cache_instance.sync_in_progress.get(
                    netuid, False
                ),
                "hotkeys_count": (
                    len(cache_data.get("hotkeys", [])) if cache_data else 0
                ),
                "last_sync_duration": (
                    cache_data.get("sync_duration", 0) if cache_data else 0
                ),
            }

    # Get info about netuids that might need syncing
    active_netuids = set()
    with cache_lock:
        for process in process_cache.get("data", []):
            if process.get("state") == "online" and process.get("netuid"):
                try:
                    active_netuids.add(str(int(process.get("netuid"))))
                except (ValueError, TypeError):
                    pass

    missing_netuids = active_netuids - set(cache_status.keys())

    return {
        "metagraph_cache_status": cache_status,
        "sync_interval_minutes": config.METAGRAPH_SYNC_INTERVAL / 60,
        "total_netuids_cached": len(cache_status),
        "active_netuids": sorted(list(active_netuids)),
        "missing_netuids": sorted(list(missing_netuids)),
        "next_scheduled_sync": (
            datetime.fromtimestamp(
                time.time()
                + max(
                    0,
                    config.METAGRAPH_SYNC_INTERVAL
                    - min(
                        [v["seconds_since_sync"] for v in cache_status.values()],
                        default=config.METAGRAPH_SYNC_INTERVAL,
                    ),
                )
            ).isoformat()
            if cache_status
            else None
        ),
    }


@app.post("/refresh")
async def force_refresh():
    """Force a manual refresh of process data"""
    logger.info("Manual refresh requested")

    # Check if already updating
    with cache_lock:
        if process_cache["is_updating"]:
            return {
                "status": "already_updating",
                "message": "Synchronization already in progress",
            }

    # Trigger async sync
    asyncio.create_task(sync_processes_async())

    return {"status": "refresh_started", "message": "Manual refresh initiated"}


@app.post("/refresh-metagraph/{netuid}")
async def force_refresh_metagraph(netuid: int):
    """Force a manual refresh of metagraph data for specific netuid"""
    logger.info(f"Manual metagraph refresh requested for netuid {netuid}")

    # Check if sync is already in progress
    with metagraph_cache_instance.lock:
        if metagraph_cache_instance.sync_in_progress.get(str(netuid), False):
            return {
                "status": "already_updating",
                "message": f"Metagraph sync already in progress for netuid {netuid}",
            }

    # Force sync
    asyncio.create_task(
        metagraph_cache_instance.get_metagraph_data_async(netuid, force=True)
    )

    return {
        "status": "refresh_started",
        "message": f"Manual metagraph refresh initiated for netuid {netuid}",
    }


@app.post("/refresh-all-metagraphs")
async def force_refresh_all_metagraphs():
    """Force a manual refresh of all active metagraphs"""
    logger.info("Manual refresh requested for all metagraphs")

    # Get active netuids
    active_netuids = set()
    with cache_lock:
        for process in process_cache.get("data", []):
            if process.get("state") == "online" and process.get("netuid"):
                try:
                    active_netuids.add(int(process.get("netuid")))
                except (ValueError, TypeError):
                    pass

    if not active_netuids:
        return {
            "status": "no_active_netuids",
            "message": "No active netuids found to refresh",
        }

    # Force sync all
    tasks = []
    for netuid in sorted(active_netuids):
        task = metagraph_cache_instance.get_metagraph_data_async(netuid, force=True)
        tasks.append(asyncio.create_task(task))

    return {
        "status": "refresh_started",
        "message": f"Manual metagraph refresh initiated for {len(active_netuids)} netuids",
        "netuids": sorted(list(active_netuids)),
    }


@app.get("/logs/{process_name}")
async def get_process_logs(
    process_name: str,
    lines: int = 50,
    log_type: str = "both",
    force_refresh: bool = False,
):
    """Get logs for a specific process"""
    if lines > 500:
        lines = 500  # Cap maximum lines

    response = {"process": process_name, "logs": {}}

    try:
        # Clear cache if force_refresh
        if force_refresh:
            with log_cache_lock:
                keys_to_remove = []
                for key in log_cache:
                    if key.startswith(f"{process_name}_"):
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del log_cache[key]

        if log_type in ["error", "both"]:
            # Override config for this specific request
            original_max_lines = config.MAX_LOG_LINES
            config.MAX_LOG_LINES = lines

            error_logs = await get_cached_logs(process_name, "error")
            response["logs"]["error"] = error_logs

            config.MAX_LOG_LINES = original_max_lines

        if log_type in ["out", "both"]:
            original_max_lines = config.MAX_LOG_LINES
            config.MAX_LOG_LINES = lines

            out_logs = await get_cached_logs(process_name, "out")
            response["logs"]["out"] = out_logs

            config.MAX_LOG_LINES = original_max_lines

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system")
async def get_system_metrics():
    """Get system-wide metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Get per-CPU usage
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)

        # Get process count by state
        process_states = {"online": 0, "stopped": 0, "errored": 0}
        with cache_lock:
            for process in process_cache.get("data", []):
                state = process.get("state", "unknown")
                if state in process_states:
                    process_states[state] += 1

        return {
            "cpu": {
                "total": cpu_percent,
                "per_core": cpu_per_core,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": memory.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent,
            },
            "processes": process_states,
            "monitor_service": get_resource_usage(),
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache-status")
async def get_cache_status():
    """Get status of all caches"""
    current_time = time.time()

    # Log cache status
    log_cache_status = {}
    with log_cache_lock:
        for key, value in log_cache.items():
            age_seconds = current_time - value.get("timestamp", 0)
            log_cache_status[key] = {
                "age_seconds": round(age_seconds),
                "age_minutes": round(age_seconds / 60, 1),
                "stale": age_seconds > config.LOG_CACHE_TTL,
                "size_bytes": len(value.get("logs", "")),
            }

    # Process cache status
    with cache_lock:
        process_cache_status = {
            "last_updated": (
                process_cache.get("last_updated").isoformat()
                if process_cache.get("last_updated")
                else None
            ),
            "is_updating": process_cache.get("is_updating", False),
            "has_error": bool(process_cache.get("error")),
            "process_count": len(process_cache.get("data", [])),
        }

    # Metagraph cache status
    metagraph_cache_status = {}
    with metagraph_cache_instance.lock:
        metagraph_cache_status = {
            "cached_netuids": len(metagraph_cache_instance.cache),
            "syncs_in_progress": sum(
                1 for v in metagraph_cache_instance.sync_in_progress.values() if v
            ),
            "total_hotkeys_cached": sum(
                len(data.get("hotkeys", []))
                for data in metagraph_cache_instance.cache.values()
            ),
        }

    return {
        "log_cache": {
            "entries": len(log_cache_status),
            "total_size_kb": round(
                sum(v["size_bytes"] for v in log_cache_status.values()) / 1024, 2
            ),
            "stale_entries": sum(1 for v in log_cache_status.values() if v["stale"]),
            "details": log_cache_status,
        },
        "process_cache": process_cache_status,
        "metagraph_cache": metagraph_cache_status,
        "cache_config": {
            "log_cache_ttl": config.LOG_CACHE_TTL,
            "metagraph_cache_ttl": config.METAGRAPH_CACHE_TTL,
            "incremental_updates": config.ENABLE_INCREMENTAL_UPDATES,
        },
    }


@app.delete("/clear-caches")
async def clear_caches():
    """Clear all caches (useful for debugging)"""
    logger.warning("Clearing all caches")

    # Clear log cache
    with log_cache_lock:
        log_cache_size = len(log_cache)
        log_cache.clear()

    # Clear metagraph cache
    with metagraph_cache_instance.lock:
        metagraph_cache_size = len(metagraph_cache_instance.cache)
        metagraph_cache_instance.cache.clear()
        metagraph_cache_instance.last_sync.clear()

    # Don't clear process cache but mark it for refresh
    asyncio.create_task(sync_processes_async())

    return {
        "status": "caches_cleared",
        "cleared": {
            "log_cache_entries": log_cache_size,
            "metagraph_cache_entries": metagraph_cache_size,
        },
        "message": "All caches cleared. Process sync triggered.",
    }


@app.get("/debug")
async def debug_info():
    """Get debug information about the service"""
    import platform

    # Get thread info
    thread_info = []
    for thread in threading.enumerate():
        thread_info.append(
            {
                "name": thread.name,
                "alive": thread.is_alive(),
                "daemon": thread.daemon,
            }
        )

    # Get asyncio task info
    tasks = asyncio.all_tasks()
    task_info = []
    for task in tasks:
        task_info.append(
            {
                "name": task.get_name() if hasattr(task, "get_name") else str(task),
                "done": task.done(),
            }
        )

    return {
        "service": {
            "version": "2.0",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "uptime_seconds": (
                time.time() - app.state.start_time
                if hasattr(app.state, "start_time")
                else None
            ),
        },
        "config": {
            "process_sync_interval": config.PROCESS_SYNC_INTERVAL,
            "metagraph_sync_interval": config.METAGRAPH_SYNC_INTERVAL,
            "adaptive_sync_enabled": config.ADAPTIVE_SYNC,
            "current_sync_interval": sync_scheduler.current_interval,
        },
        "threads": {
            "count": threading.active_count(),
            "list": thread_info,
        },
        "async_tasks": {
            "count": len(task_info),
            "list": task_info,
        },
        "resource_usage": get_resource_usage(),
    }


@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    app.state.start_time = time.time()
    logger.info(f"PM2 Process Monitor v2.0 started at {datetime.now().isoformat()}")
    logger.info(f"Configuration: {config}")


if __name__ == "__main__":
    logger.info("PM2 Process Monitor starting on http://127.0.0.1:8080/")
    logger.info(f"Process sync interval: {config.PROCESS_SYNC_INTERVAL}s")
    logger.info(f"Metagraph sync interval: {config.METAGRAPH_SYNC_INTERVAL}s (strict)")
    logger.info(f"Adaptive sync: {'enabled' if config.ADAPTIVE_SYNC else 'disabled'}")

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
