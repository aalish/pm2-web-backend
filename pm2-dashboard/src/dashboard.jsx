import React, { useState, useEffect } from "react";
import PropTypes from "prop-types";
import {
  Box,
  Collapse,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Paper,
  TextField,
  Button,
  Tabs,
  Tab,
  CircularProgress,
  Chip,
  Tooltip,
  styled,
  Alert,
  LinearProgress,
  Grid,
  Card,
  CardContent,
} from "@mui/material";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CheckIcon from "@mui/icons-material/Check";
import CircleIcon from "@mui/icons-material/Circle";
import MemoryIcon from "@mui/icons-material/Memory";
import SpeedIcon from "@mui/icons-material/Speed";
import StorageIcon from "@mui/icons-material/Storage";
import ErrorIcon from "@mui/icons-material/Error";
import WarningIcon from "@mui/icons-material/Warning";

// Custom styled components
const StyledTableCell = styled(TableCell)(({ theme }) => ({
  fontFamily: "'Inter', 'Roboto', sans-serif",
  fontSize: "0.875rem",
  padding: "8px 12px",
  "&.MuiTableCell-head": {
    fontWeight: 600,
    fontSize: "0.8125rem",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    color: theme.palette.text.secondary,
  },
}));

const CompactChip = styled(Chip)(({ theme }) => ({
  height: "22px",
  fontSize: "0.75rem",
  fontWeight: 500,
}));

// Resource monitoring card
const ResourceCard = styled(Card)(({ theme }) => ({
  minHeight: "80px",
  position: "relative",
  overflow: "visible",
}));

const ResourceProgress = styled(LinearProgress)(({ theme, severity }) => ({
  height: 8,
  borderRadius: 4,
  backgroundColor: theme.palette.grey[200],
  "& .MuiLinearProgress-bar": {
    borderRadius: 4,
    ...(severity === "danger" && {
      backgroundColor: theme.palette.error.main,
    }),
    ...(severity === "warning" && {
      backgroundColor: theme.palette.warning.main,
    }),
    ...(severity === "healthy" && {
      backgroundColor: theme.palette.success.main,
    }),
  },
}));

// Helper to determine resource severity
const getResourceSeverity = (type, value) => {
  if (type === "memory") {
    if (value > 1024) return "danger"; // > 1GB
    if (value > 768) return "warning"; // > 768MB
    return "healthy";
  }
  if (type === "cpu") {
    if (value > 80) return "danger"; // > 80%
    if (value > 60) return "warning"; // > 60%
    return "healthy";
  }
  return "healthy";
};

// Resource Monitor Component
function ResourceMonitor({ stats }) {
  const resources = stats?.resources || {};
  const memoryMB = resources.memory_mb || 0;
  const cpuPercent = resources.cpu_percent || 0;
  const threads = resources.threads || 0;
  const openFiles = resources.open_files || 0;
  const syncInterval = stats?.sync_interval || 60;

  const memorySeverity = getResourceSeverity("memory", memoryMB);
  const cpuSeverity = getResourceSeverity("cpu", cpuPercent);

  return (
    <Box
      sx={{
        p: 2,
        backgroundColor: "#f8f9fa",
        borderBottom: "1px solid #e0e0e0",
      }}
    >
      <Typography
        variant="subtitle2"
        sx={{ mb: 2, fontWeight: 600, color: "text.secondary" }}
      >
        PM2 Monitor Service Resources
      </Typography>

      <Grid container spacing={2}>
        {/* Memory Usage */}
        <Grid item xs={12} sm={6} md={3}>
          <ResourceCard variant="outlined">
            <CardContent sx={{ p: 1.5 }}>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}
              >
                <MemoryIcon
                  sx={{
                    fontSize: 20,
                    color:
                      memorySeverity === "danger"
                        ? "error.main"
                        : "text.secondary",
                  }}
                />
                <Typography variant="body2" color="text.secondary">
                  Memory Usage
                </Typography>
                {memorySeverity === "danger" && (
                  <Tooltip title="High memory usage detected!">
                    <ErrorIcon
                      sx={{ fontSize: 16, color: "error.main", ml: "auto" }}
                    />
                  </Tooltip>
                )}
                {memorySeverity === "warning" && (
                  <Tooltip title="Memory usage is elevated">
                    <WarningIcon
                      sx={{ fontSize: 16, color: "warning.main", ml: "auto" }}
                    />
                  </Tooltip>
                )}
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                {memoryMB.toFixed(0)} MB
              </Typography>
              <ResourceProgress
                variant="determinate"
                value={Math.min((memoryMB / 1024) * 100, 100)}
                severity={memorySeverity}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.5, display: "block" }}
              >
                {((memoryMB / 1024) * 100).toFixed(1)}% of 1GB threshold
              </Typography>
            </CardContent>
          </ResourceCard>
        </Grid>

        {/* CPU Usage */}
        <Grid item xs={12} sm={6} md={3}>
          <ResourceCard variant="outlined">
            <CardContent sx={{ p: 1.5 }}>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}
              >
                <SpeedIcon
                  sx={{
                    fontSize: 20,
                    color:
                      cpuSeverity === "danger"
                        ? "error.main"
                        : "text.secondary",
                  }}
                />
                <Typography variant="body2" color="text.secondary">
                  CPU Usage
                </Typography>
                {cpuSeverity === "danger" && (
                  <Tooltip title="High CPU usage detected!">
                    <ErrorIcon
                      sx={{ fontSize: 16, color: "error.main", ml: "auto" }}
                    />
                  </Tooltip>
                )}
                {cpuSeverity === "warning" && (
                  <Tooltip title="CPU usage is elevated">
                    <WarningIcon
                      sx={{ fontSize: 16, color: "warning.main", ml: "auto" }}
                    />
                  </Tooltip>
                )}
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
                {cpuPercent.toFixed(1)}%
              </Typography>
              <ResourceProgress
                variant="determinate"
                value={cpuPercent}
                severity={cpuSeverity}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.5, display: "block" }}
              >
                Across 4 CPU cores
              </Typography>
            </CardContent>
          </ResourceCard>
        </Grid>

        {/* Threads & Files */}
        <Grid item xs={12} sm={6} md={3}>
          <ResourceCard variant="outlined">
            <CardContent sx={{ p: 1.5 }}>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}
              >
                <StorageIcon sx={{ fontSize: 20, color: "text.secondary" }} />
                <Typography variant="body2" color="text.secondary">
                  System Info
                </Typography>
              </Box>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="caption" color="text.secondary">
                    Threads:
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {threads}
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="caption" color="text.secondary">
                    Open Files:
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {openFiles}
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="caption" color="text.secondary">
                    Processes:
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {stats?.process_count || 0}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </ResourceCard>
        </Grid>

        {/* Sync Status */}
        <Grid item xs={12} sm={6} md={3}>
          <ResourceCard variant="outlined">
            <CardContent sx={{ p: 1.5 }}>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}
              >
                <CircleIcon sx={{ fontSize: 10, color: "success.main" }} />
                <Typography variant="body2" color="text.secondary">
                  Sync Status
                </Typography>
              </Box>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="caption" color="text.secondary">
                    Interval:
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {syncInterval}s
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                  <Typography variant="caption" color="text.secondary">
                    Mode:
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {syncInterval > 60 ? "Idle" : "Active"}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </ResourceCard>
        </Grid>
      </Grid>

      {/* Alert for high resource usage */}
      {(memorySeverity === "danger" || cpuSeverity === "danger") && (
        <Alert severity="error" sx={{ mt: 2 }} icon={<ErrorIcon />}>
          <strong>High Resource Usage Detected!</strong>
          {memorySeverity === "danger" &&
            ` Memory usage (${memoryMB.toFixed(0)} MB) exceeds 1GB threshold.`}
          {cpuSeverity === "danger" &&
            ` CPU usage (${cpuPercent.toFixed(1)}%) is critically high.`}{" "}
          Consider restarting the service or investigating the cause.
        </Alert>
      )}

      {(memorySeverity === "warning" || cpuSeverity === "warning") &&
        !(memorySeverity === "danger" || cpuSeverity === "danger") && (
          <Alert severity="warning" sx={{ mt: 2 }} icon={<WarningIcon />}>
            <strong>Elevated Resource Usage</strong>
            {memorySeverity === "warning" &&
              ` Memory: ${memoryMB.toFixed(0)} MB.`}
            {cpuSeverity === "warning" && ` CPU: ${cpuPercent.toFixed(1)}%.`}{" "}
            Monitor for further increases.
          </Alert>
        )}
    </Box>
  );
}

// Update the main component
export default function CollapsibleTable() {
  const [rows, setRows] = useState([]);
  const [auth, setAuth] = useState({ username: "", password: "" });
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isloading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [stats, setStats] = useState({}); // Add stats state
  // Update fetchData to capture stats
  const fetchData = async (authData = auth) => {
    try {
      const basicAuth =
        "Basic " + btoa(`${authData.username}:${authData.password}`);

      const res = await fetch(process.env.REACT_APP_BACKEND_URL, {
        headers: {
          Authorization: basicAuth,
        },
      });

      if (res.status !== 200) {
        setError("Invalid username or password.");
        setIsAuthenticated(false);
        localStorage.removeItem("pm2_auth");
        return;
      }

      if (!res.ok) throw new Error("Failed to fetch");

      const response = await res.json();

      // Check if response has error
      if (response.error) {
        setError(`Server error: ${response.error}`);
        return;
      }

      // Sort data by state (online first)
      const sorted = response.data.sort((a, b) => {
        if (a.state === "online" && b.state !== "online") return -1;
        if (a.state !== "online" && b.state === "online") return 1;
        return 0;
      });

      setRows(sorted);
      setLastUpdated(response.last_updated);
      setIsUpdating(response.is_updating);
      setStats(response.stats || {}); // Capture stats
      setIsAuthenticated(true);
      setError("");

      // Store credentials on successful login
      localStorage.setItem("pm2_auth", JSON.stringify(authData));
    } catch (err) {
      console.error("Fetch error:", err);
      setError("Failed to fetch data. Please try again.");
    } finally {
      setIsLoading(false);
      setIsCheckingAuth(false);
    }
  };

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (isAuthenticated) {
      const interval = setInterval(() => {
        fetchData();
      }, 30000);
      return () => clearInterval(interval);
    }
    setIsCheckingAuth(false);
  }, [isAuthenticated]);

  const handleLogin = () => {
    setIsLoading(true);
    if (auth.username && auth.password) {
      fetchData();
    } else {
      setError("Please enter both username and password.");
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("pm2_auth");
    setIsAuthenticated(false);
    setAuth({ username: "", password: "" });
    setRows([]);
    setStats({});
    setError("");
  };

  const formatDateTime = (dateString) => {
    if (!dateString) return "";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Show loading while checking stored auth
  if (isCheckingAuth) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (!isAuthenticated) {
    return (
      <Box
        sx={{
          p: 4,
          maxWidth: 400,
          margin: "auto",
          mt: 8,
          border: "1px solid #ccc",
          borderRadius: 2,
        }}
      >
        <Typography variant="h6" gutterBottom>
          Login
        </Typography>
        <TextField
          label="Username"
          variant="outlined"
          fullWidth
          margin="normal"
          value={auth.username}
          onChange={(e) =>
            setAuth((prev) => ({ ...prev, username: e.target.value }))
          }
          onKeyPress={(e) => e.key === "Enter" && handleLogin()}
        />
        <TextField
          label="Password"
          type="password"
          variant="outlined"
          fullWidth
          margin="normal"
          value={auth.password}
          onChange={(e) =>
            setAuth((prev) => ({ ...prev, password: e.target.value }))
          }
          onKeyPress={(e) => e.key === "Enter" && handleLogin()}
        />
        {error && (
          <Typography color="error" variant="body2" sx={{ mt: 1 }}>
            {error}
          </Typography>
        )}
        <Button
          variant="contained"
          color="primary"
          onClick={handleLogin}
          sx={{ mt: 2 }}
          fullWidth
          disabled={isloading}
        >
          {isloading ? "Logging In..." : "Login"}
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ fontFamily: "'Inter', 'Roboto', sans-serif" }}>
      {/* Compact Header */}
      <Box
        sx={{
          p: 1.5,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          backgroundColor: "#f8f9fa",
          borderBottom: "1px solid #e0e0e0",
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontSize: "1.125rem",
            fontWeight: 600,
            letterSpacing: "-0.5px",
          }}
        >
          PM2 Process Monitor
        </Typography>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          {isUpdating && (
            <Chip
              label="Updating..."
              size="small"
              color="primary"
              sx={{ height: "24px", fontSize: "0.75rem" }}
            />
          )}
          {lastUpdated && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: "0.75rem" }}
            >
              Updated: {formatDateTime(lastUpdated)}
            </Typography>
          )}
          <Button
            variant="outlined"
            size="small"
            onClick={() => fetchData()}
            disabled={isloading}
            sx={{
              height: "32px",
              fontSize: "0.8125rem",
              textTransform: "none",
            }}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            size="small"
            color="error"
            onClick={handleLogout}
            sx={{
              height: "32px",
              fontSize: "0.8125rem",
              textTransform: "none",
            }}
          >
            Logout
          </Button>
        </Box>
      </Box>

      {/* Resource Monitor Section */}
      <ResourceMonitor stats={stats} />

      {error && (
        <Box
          sx={{
            p: 1.5,
            backgroundColor: "#fff3cd",
            borderBottom: "1px solid #ffeaa7",
          }}
        >
          <Typography color="error" sx={{ fontSize: "0.875rem" }}>
            {error}
          </Typography>
        </Box>
      )}

      <TableContainer component={Paper} elevation={0}>
        <Table size="small" aria-label="collapsible table">
          <TableHead>
            <TableRow sx={{ backgroundColor: "#f8f9fa" }}>
              <StyledTableCell />
              <StyledTableCell sx={{ width: "15px" }}>ID</StyledTableCell>
              <StyledTableCell align="center">Name</StyledTableCell>
              <StyledTableCell align="center">Status</StyledTableCell>
              <StyledTableCell align="center">NetUID</StyledTableCell>
              <StyledTableCell align="center">UID</StyledTableCell>
              <StyledTableCell>Wallet</StyledTableCell>
              <StyledTableCell>Hotkey</StyledTableCell>
              <StyledTableCell sx={{ width: "150px" }}>
                Hotkey SS58
              </StyledTableCell>
              <StyledTableCell align="center">Miner UID</StyledTableCell>
              <StyledTableCell align="center">Stake</StyledTableCell>
              <StyledTableCell align="center">Emission</StyledTableCell>
              <StyledTableCell align="center">Daily Alpha</StyledTableCell>
              <StyledTableCell align="center">Port</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row, index) => (
              <Row key={`${row.name}-${row.id}`} row={row} index={index} />
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

// Update Row component (keep the original Row component as is)
function Row({ row, index }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [logTab, setLogTab] = useState(0);

  const handleCopy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const isOnline = row.state === "online";

  return (
    <>
      <TableRow
        sx={{
          "& > *": { borderBottom: "unset" },
          backgroundColor: index % 2 === 0 ? "#fafafa" : "#ffffff",
          "&:hover": {
            backgroundColor: "#f5f5f5",
            transition: "background-color 0.2s",
          },
          opacity: isOnline ? 1 : 0.7,
        }}
      >
        <StyledTableCell sx={{ width: "40px" }}>
          <IconButton size="small" onClick={() => setOpen(!open)}>
            {open ? (
              <KeyboardArrowUpIcon fontSize="small" />
            ) : (
              <KeyboardArrowDownIcon fontSize="small" />
            )}
          </IconButton>
        </StyledTableCell>

        <StyledTableCell>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <CircleIcon
              sx={{
                fontSize: 10,
                color: isOnline ? "success.main" : "error.main",
              }}
            />
            <Typography
              variant="body2"
              sx={{ fontFamily: "monospace", fontWeight: 500 }}
            >
              {row.id}
            </Typography>
          </Box>
        </StyledTableCell>

        <StyledTableCell align="center">{row.name}</StyledTableCell>
        <StyledTableCell align="center">
          <CompactChip
            label={row.state}
            color={isOnline ? "success" : "error"}
            size="small"
            variant="outlined"
          />
        </StyledTableCell>

        <StyledTableCell align="center">{row.netuid}</StyledTableCell>

        <StyledTableCell align="center">
          <Tooltip title={`UID: ${row.uid}`}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {row.uid === -1 ? "-" : row.uid}
            </Typography>
          </Tooltip>
        </StyledTableCell>

        <StyledTableCell>{row.wallet_name}</StyledTableCell>
        <StyledTableCell>{row.hotkey}</StyledTableCell>
        <StyledTableCell>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Tooltip title={row.hotkey_ss58 || "No hotkey"}>
              <Typography
                variant="body2"
                sx={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "0.8125rem",
                  cursor: row.hotkey ? "pointer" : "default",
                  whiteSpace: "nowrap",
                }}
              >
                {row.hotkey_ss58
                  ? `${row.hotkey_ss58.slice(0, 6)}...${row.hotkey_ss58.slice(
                      -4
                    )}`
                  : "-"}
              </Typography>
            </Tooltip>
            {row.hotkey_ss58 && (
              <IconButton
                size="small"
                onClick={() => handleCopy(row.hotkey_ss58)}
                sx={{ padding: "2px" }}
              >
                {copied ? (
                  <CheckIcon sx={{ fontSize: 14, color: "success.main" }} />
                ) : (
                  <ContentCopyIcon sx={{ fontSize: 14 }} />
                )}
              </IconButton>
            )}
          </Box>
        </StyledTableCell>

        <StyledTableCell align="center">{row.miner_uid || "-"}</StyledTableCell>
        <StyledTableCell align="center">
          {Math.round(row.stake * 100) / 100 || 0}
        </StyledTableCell>
        <StyledTableCell align="center">
          {Math.round(row.emission * 10 ** 5) / 10 ** 5 || 0}
        </StyledTableCell>
        <StyledTableCell align="center">
          {Math.round(row.daily_alpha * 10 ** 2) / 10 ** 2 || 0}
        </StyledTableCell>

        <StyledTableCell align="center">
          <Typography
            variant="body2"
            sx={{ fontFamily: "monospace", fontSize: "0.8125rem" }}
          >
            {row.port || "-"}
          </Typography>
        </StyledTableCell>
      </TableRow>

      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={14}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <Tabs
                value={logTab}
                onChange={(e, newValue) => setLogTab(newValue)}
                sx={{
                  minHeight: "36px",
                  "& .MuiTab-root": {
                    minHeight: "36px",
                    fontSize: "0.8125rem",
                    textTransform: "none",
                  },
                }}
              >
                <Tab label="Error Logs" />
                <Tab label="Output Logs" />
              </Tabs>

              <Box
                component="pre"
                sx={{
                  whiteSpace: "pre-wrap",
                  backgroundColor: "#1e1e1e",
                  color: "#d4d4d4",
                  padding: 1.5,
                  borderRadius: 1,
                  maxHeight: 250,
                  overflow: "auto",
                  fontSize: "0.75rem",
                  fontFamily: "'JetBrains Mono', 'Consolas', monospace",
                  mt: 1,
                  "& ::-webkit-scrollbar": {
                    width: "8px",
                  },
                  "& ::-webkit-scrollbar-track": {
                    background: "#2e2e2e",
                  },
                  "& ::-webkit-scrollbar-thumb": {
                    background: "#555",
                    borderRadius: "4px",
                  },
                }}
              >
                {logTab === 0
                  ? row.error_logs || "No error logs available."
                  : row.out_logs || "No output logs available."}
              </Box>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

Row.propTypes = {
  row: PropTypes.shape({
    id: PropTypes.number.isRequired,
    name: PropTypes.string.isRequired,
    state: PropTypes.string.isRequired,
    uid: PropTypes.number.isRequired,
    netuid: PropTypes.string.isRequired,
    wallet_name: PropTypes.string.isRequired,
    hotkey: PropTypes.string.isRequired,
    hotkey_ss58: PropTypes.string,
    miner_uid: PropTypes.string,
    port: PropTypes.string,
    stake: PropTypes.number,
    emission: PropTypes.number,
    daily_alpha: PropTypes.number,
    error_logs: PropTypes.string,
    out_logs: PropTypes.string,
  }).isRequired,
  index: PropTypes.number.isRequired,
};
