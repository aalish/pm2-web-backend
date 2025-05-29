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
} from "@mui/material";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CheckIcon from "@mui/icons-material/Check";
import CircleIcon from "@mui/icons-material/Circle";

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
          <Typography
            variant="body2"
            sx={{ fontFamily: "monospace", fontSize: "0.8125rem" }}
          >
            {row.port || "-"}
          </Typography>
        </StyledTableCell>
      </TableRow>

      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={9}>
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
    coldkey: PropTypes.string,
    miner_uid: PropTypes.string,
    port: PropTypes.string,
    error_logs: PropTypes.string,
    out_logs: PropTypes.string,
  }).isRequired,
  index: PropTypes.number.isRequired,
};

export default function CollapsibleTable() {
  const [rows, setRows] = useState([]);
  const [auth, setAuth] = useState({ username: "", password: "" });
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isloading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  // Check for stored credentials on mount
  useEffect(() => {
    const storedAuth = localStorage.getItem("pm2_auth");
    if (storedAuth) {
      try {
        const parsedAuth = JSON.parse(storedAuth);
        setAuth(parsedAuth);
        // Attempt auto-login with stored credentials
        fetchData(parsedAuth);
      } catch (err) {
        console.error("Failed to parse stored auth:", err);
        localStorage.removeItem("pm2_auth");
        setIsCheckingAuth(false);
      }
    } else {
      setIsCheckingAuth(false);
    }
  }, []);

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
        // Clear stored credentials if invalid
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
