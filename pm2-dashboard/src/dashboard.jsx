import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
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
  Button
} from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

function createData(process, state, netuid, wallet, hotkey, minerUid, port, errorLogs) {
  return { process, state, netuid, wallet, hotkey, minerUid, port, errorLogs };
}

function Row({ row, index }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <TableRow
        sx={{
          '& > *': { borderBottom: 'unset' },
          backgroundColor: index % 2 === 0 ? '#f9f9f9' : '#ffffff',
        }}
      >
        <TableCell>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => setOpen(!open)}
          >
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell component="th" scope="row">{row.process}</TableCell>
        <TableCell align="right">{row.state}</TableCell>
        <TableCell align="right">{row.netuid}</TableCell>
        <TableCell align="right">{row.walletName}</TableCell>
        <TableCell align="right">{row.hotkey}</TableCell>
        <TableCell align="right">{row.minerUid}</TableCell>
        <TableCell align="right">{row.axonPort}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={8}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <Typography variant="subtitle1" gutterBottom component="div">
                Error Logs
              </Typography>
              <Box
                component="pre"
                sx={{
                  whiteSpace: 'pre-wrap',
                  backgroundColor: '#f5f5f5',
                  padding: 1,
                  borderRadius: 1,
                  maxHeight: 300,
                  overflow: 'auto',
                }}
              >
                {row.errorLogs || 'No error logs available.'}
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
    process: PropTypes.string.isRequired,
    state: PropTypes.string.isRequired,
    netuid: PropTypes.string.isRequired,
    walletName: PropTypes.string.isRequired,
    hotkey: PropTypes.string.isRequired,
    minerUid: PropTypes.string,
    axonPort: PropTypes.string,
    errorLogs: PropTypes.string,
  }).isRequired,
  index: PropTypes.number.isRequired,
};

export default function CollapsibleTable() {
  const [rows, setRows] = useState([]);
  const [auth, setAuth] = useState({ username: '', password: '' });
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isloading,setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchData = async () => {
    try {
      const basicAuth = 'Basic ' + btoa(`${auth.username}:${auth.password}`);

      const res = await fetch('https://bpm2.codezilla.solutions', {
        headers: {
          Authorization: basicAuth,
        },
      });
      console.log(res)
      if (res.status!==200) {
        setError('Invalid username or password.');
        setIsAuthenticated(false);
        return;
      }
      if (!res.ok) throw new Error('Failed to fetch');

      const data = await res.json();

      const parsed = Object.entries(data).map(([process, { State = '', Args = [], 'Error logs': errorLogs = '' }]) => {
        const argMap = Args.reduce((m, v, i, arr) => {
          if (v.startsWith('--')) {
            const key = v.slice(2);
            const next = arr[i + 1];
            m[key] = next && !next.startsWith('--') ? next : true;
          }
          return m;
        }, {});

        return {
          process,
          state: State,
          netuid: argMap.netuid || '',
          walletName: argMap['wallet.name'] || '',
          hotkey: argMap['wallet.hotkey'] || '',
          minerUid: argMap['miner_uid'] || argMap['miner.uid'] || '',
          axonPort: argMap['axon.port'] || '8091',
          errorLogs
        };
      });

      const sorted = parsed.sort((a, b) => {
        if (a.state === 'online' && b.state !== 'online') return -1;
        if (a.state !== 'online' && b.state === 'online') return 1;
        return 0;
      });

      setRows(sorted);
      setIsAuthenticated(true);
      setError('');
    } catch (err) {
      console.error('Fetch error:', err);
      setError('Invalid username or password.');
    }
    setIsLoading(false);
  };

  const handleLogin = () => {
    setIsLoading(true)
    if (auth.username && auth.password) {
      fetchData();
    } else {
      setError('Please enter both username and password.');
    }
  };

  if (!isAuthenticated) {
    return (
      <Box sx={{ p: 4, maxWidth: 400, margin: 'auto', mt: 8, border: '1px solid #ccc', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom>Login</Typography>
        <TextField
          label="Username"
          variant="outlined"
          fullWidth
          margin="normal"
          value={auth.username}
          onChange={(e) => setAuth((prev) => ({ ...prev, username: e.target.value }))}
        />
        <TextField
          label="Password"
          type="password"
          variant="outlined"
          fullWidth
          margin="normal"
          value={auth.password}
          onChange={(e) => setAuth((prev) => ({ ...prev, password: e.target.value }))}
        />
        {error && <Typography color="error" variant="body2">{error}</Typography>}
        <Button
          variant="contained"
          color="primary"
          onClick={handleLogin}
          sx={{ mt: 2 }}
        >
          {isloading?"Logging In":"Login"}
        </Button>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table aria-label="collapsible table">
        <TableHead>
          <TableRow sx={{ backgroundColor: '#d0f0c0' }}>
            <TableCell />
            <TableCell>Process</TableCell>
            <TableCell align="right">State</TableCell>
            <TableCell align="right">NetUID</TableCell>
            <TableCell align="right">Wallet</TableCell>
            <TableCell align="right">Hotkey</TableCell>
            <TableCell align="right">Miner UID</TableCell>
            <TableCell align="right">Port</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row, index) => (
            <Row key={row.process} row={row} index={index} />
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
