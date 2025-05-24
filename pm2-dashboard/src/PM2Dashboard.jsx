import React, { useEffect, useState } from 'react';

const fetchData = async () => {
  const res = await fetch('http://localhost:8080');
  if (!res.ok) throw new Error('Failed to fetch');
  return res.json();
};

const PM2Dashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    fetchData()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const toggleExpand = (name) => {
    setExpanded((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  if (loading) return <div className="container">Loading...</div>;
  if (error) return <div className="container error">Error: {error}</div>;

  return (
    <div className="container">
      {Object.entries(data).map(([name, info]) => (
        <div className="card" key={name}>
          <div className="card-header">
            <h4>{name}</h4>
            <span className={`status ${info.State === 'online' ? 'online' : 'offline'}`}>
              {info.State}
            </span>
          </div>

          <div className="card-section">
            <strong>Args:</strong>
            <code>{info.Args.join(' ')}</code>
          </div>

          <div className="card-section">
            <button className="toggle-btn" onClick={() => toggleExpand(name)}>
              {expanded[name] ? 'Hide Error Logs' : 'Show Error Logs'}
            </button>
            {expanded[name] && (
              <pre className="log">{info['Error logs']}</pre>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default PM2Dashboard;
