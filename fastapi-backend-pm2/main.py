from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import subprocess
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
import typing as ty

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

process_cache = {"data": [], "last_updated": None, "is_updating": False, "error": None}

# Lock for thread-safe access to cache
cache_lock = threading.Lock()


# Define models
class ProcessInfo:
    def __init__(
        self, id: int, state: str, args: List[str], error_logs: str, out_logs: str
    ):
        self.id = id
        self.State = state
        self.Args = args
        self.ErrorLogs = error_logs
        self.outLogs = out_logs

    def dict(self):
        return {
            "Id": self.id,
            "State": self.State,
            "Args": self.Args,
            "Error logs": self.ErrorLogs,
            "Out logs": self.outLogs,
        }


class ResponseData:
    def __init__(
        self,
        id: int,
        name: str,
        state: str,
        uid: int,
        netuid: str,
        wallet_name: str,
        hotkey: str,
        hotkey_ss58: str,
        miner_uid: str,
        port: str,
        error_logs: str,
        out_logs: str,
    ):
        self.id = id
        self.name = name
        self.state = state
        self.uid = uid
        self.netuid = netuid
        self.wallet_name = wallet_name
        self.hotkey = hotkey
        self.hotkey_ss58 = hotkey_ss58
        self.miner_uid = miner_uid
        self.port = port
        self.error_logs = error_logs
        self.out_logs = out_logs

    def dict(self):
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
            "error_logs": self.error_logs,
            "out_logs": self.out_logs,
        }


class ErrorResponse:
    def __init__(self, id: int, error_logs: str, state: str, error: str, name: str):
        self.id = id
        self.error = error
        self.state = state
        self.error_logs = error_logs
        self.name = name

    def dict(self):
        return {
            "id": self.id,
            "error": self.error,
            "state": self.state,
            "error_logs": self.error_logs,
            "name": self.name,
        }


def extract_arg_value(args: List[str], arg_name: str) -> Optional[str]:
    """Extract value for a specific argument from args list"""
    try:
        # Find the index of the argument
        idx = args.index(arg_name)
        # Return the next value if it exists
        if idx + 1 < len(args):
            return args[idx + 1]
    except ValueError:
        pass
    return None


def get_pm2_processes() -> Dict[str, ProcessInfo]:
    """Get all PM2 processes and their information"""
    try:
        # Run pm2 jlist command
        result = subprocess.run(
            ["pm2", "jlist"], capture_output=True, text=True, check=True
        )

        # Parse JSON output
        processes = json.loads(result.stdout)

        process_info = {}

        for process in processes:
            id = process.get("pm_id", "")
            name = process.get("name", "")
            status = process.get("pm2_env", {}).get("status", "unknown")
            args = process.get("pm2_env", {}).get("args", [])

            # Get error logs for this process
            error_logs = get_error_logs(name)
            out_logs = get_out_logs(name)

            process_info[name] = ProcessInfo(
                id=id, state=status, args=args, error_logs=error_logs, out_logs=out_logs
            )

        return process_info

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run pm2 jlist: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get PM2 processes: {str(e)}"
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse pm2 jlist output: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse PM2 output")


def get_error_logs(name: str) -> str:
    """Get error logs for a specific PM2 process"""
    try:
        # Run pm2 logs command
        result = subprocess.run(
            ["pm2", "logs", name, "--lines", "100", "--err", "--raw", "--nostream"],
            capture_output=True,
            text=True,
            timeout=10,
        )  # Add timeout to prevent hanging

        # Combine stdout and stderr
        output = result.stdout + result.stderr

        # Strip ANSI escape sequences
        cleaned_output = strip_ansi(output)

        return cleaned_output

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout getting logs for {name}")
        return "Error fetching logs: timeout"
    except Exception as e:
        logger.warning(f"Failed to get error logs for {name}: {e}")
        return "Error fetching logs"


def get_out_logs(name: str) -> str:
    """Get error logs for a specific PM2 process"""
    try:
        # Run pm2 logs command
        result = subprocess.run(
            ["pm2", "logs", name, "--lines", "100", "--raw", "--nostream"],
            capture_output=True,
            text=True,
            timeout=10,
        )  # Add timeout to prevent hanging

        # Combine stdout and stderr
        output = result.stdout + result.stderr

        # Strip ANSI escape sequences
        cleaned_output = strip_ansi(output)

        return cleaned_output

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout getting logs for {name}")
        return "Error fetching logs: timeout"
    except Exception as e:
        logger.warning(f"Failed to get error logs for {name}: {e}")
        return "Error fetching logs"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text"""
    # Regex pattern to match ANSI escape sequences
    ansi_escape = re.compile(
        r"""
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by parameter bytes
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    """,
        re.VERBOSE,
    )

    return ansi_escape.sub("", text)


def sync_processes():
    """Synchronize process data - runs in background thread"""
    global process_cache

    logger.info("Starting process synchronization...")

    try:
        with cache_lock:
            process_cache["is_updating"] = True
            process_cache["error"] = None

        processes = get_pm2_processes()

        # Extract unique netuids and their associated wallet names
        netuid_wallets = defaultdict(set)
        netuid_hotkeys = defaultdict(list)

        # Convert to dictionary format for JSON response
        response_processes = {}
        for name, info in processes.items():
            response_processes[name] = info.dict()
            # Extract netuid and wallet.name from args
            netuid = extract_arg_value(info.Args, "--netuid")
            wallet_name = extract_arg_value(info.Args, "--wallet.name")

            if netuid and wallet_name and info.State == "online":
                netuid_wallets[netuid].add(wallet_name)

        # Sync metagraphs for each netuid
        netuids = list(netuid_wallets.keys())
        for netuid in netuids:
            try:
                logger.info(f"Syncing metagraph for netuid {netuid}")
                metagraph = bt.metagraph(netuid=int(netuid))
                metagraph.sync()
                netuid_hotkeys[netuid] = metagraph.hotkeys
            except Exception as e:
                logger.error(f"Failed to sync metagraph for netuid {netuid}: {e}")
                netuid_hotkeys[netuid] = []

        # Build response data
        responses: ty.List[ResponseData | ErrorResponse] = []
        for name, info in processes.items():
            try:
                netuid = extract_arg_value(info.Args, "--netuid")
                wallet_name = extract_arg_value(info.Args, "--wallet.name")
                hotkey = extract_arg_value(info.Args, "--wallet.hotkey")
                miner_uid = extract_arg_value(info.Args, "--miner.uid")
                port = extract_arg_value(info.Args, "--axon.port")

                if netuid and wallet_name and hotkey:
                    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
                    # Try to find UID
                    uid = -1
                    if netuid in netuid_hotkeys and netuid_hotkeys[netuid]:
                        try:
                            # Using the hardcoded address for now as in your example
                            # uid = netuid_hotkeys[netuid].index(
                            #     "5EsNzkZ3DwDqCsYmSJDeGXX51dQJd5broUCH6dbDjvkTcicD"
                            # )
                            # Uncomment below to use actual wallet hotkey
                            uid = netuid_hotkeys[netuid].index(
                                wallet.hotkey.ss58_address
                            )
                        except ValueError:
                            logger.warning(f"Hotkey not found in metagraph for {name}")
                            uid = -1
                    data = ResponseData(
                        id=info.id,
                        name=name,
                        state=info.State,
                        uid=uid,
                        netuid=netuid,
                        wallet_name=wallet_name,
                        hotkey=hotkey,
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        miner_uid=miner_uid or "",
                        port=port or "",
                        error_logs=info.ErrorLogs,
                        out_logs=info.outLogs,
                    )
                    responses.append(data)
            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                # Add basic info even if processing fails
                data = ErrorResponse(
                    id=info.id,
                    name=name,
                    state=info.State,
                    error=str(e),
                    error_logs=info.ErrorLogs,
                )
                responses.append(data)

        # Update cache with new data
        with cache_lock:
            process_cache["data"] = [resp.dict() for resp in responses]
            process_cache["last_updated"] = datetime.now()
            process_cache["is_updating"] = False
            process_cache["error"] = None

        logger.info(
            f"Process synchronization completed. Found {len(responses)} processes."
        )

    except Exception as e:
        logger.error(f"Error during synchronization: {e}")
        with cache_lock:
            process_cache["is_updating"] = False
            process_cache["error"] = str(e)


def background_sync_loop():
    """Background thread that runs synchronization every 30 seconds"""
    while True:
        try:
            sync_processes()
        except Exception as e:
            logger.error(f"Unexpected error in sync loop: {e}")

        # Wait 30 seconds before next sync
        time.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global background_thread_running

    # Startup
    logger.info("Starting up...")

    # Run initial sync
    sync_processes()

    # Start background sync loop
    background_thread_running = True
    loop_thread = threading.Thread(target=background_sync_loop, daemon=True)
    loop_thread.start()

    logger.info("Background synchronization started")

    yield

    # Shutdown
    logger.info("Shutting down...")
    background_thread_running = False
    # Give the background thread time to stop gracefully
    time.sleep(1)
    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

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
    }

    # If no data yet and not updating, trigger a sync
    if not cache_data["data"] and not cache_data["is_updating"]:
        logger.info("No cached data available, triggering sync...")
        return {
            "data": [],
            "last_updated": None,
            "is_updating": True,
            "error": None,
            "message": "Initial synchronization in progress, please retry in a few seconds",
        }

    return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info("Server starting on http://127.0.0.1:8080/")
    uvicorn.run("main:app", host="127.0.0.1", port=8080)
