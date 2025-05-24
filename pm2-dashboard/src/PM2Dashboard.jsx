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

  useEffect(() => {
    fetchData()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="container">Loading...</div>;
  if (error) return <div className="container error">Error: {error}</div>;

  return (
    <div className="container">
      {Object.entries(data).map(([name, info]) => (
        <div className="card" key={name}>
          <div className="card-header">
            <h2>{name}</h2>
            <span className={`status ${info.State === 'online' ? 'online' : 'offline'}`}>
              {info.State}
            </span>
          </div>
          <div className="card-section">
            <strong>Args:</strong>
            <pre>{JSON.stringify(info.Args, null, 2)}</pre>
          </div>
          <div className="card-section">
            <strong>Error Logs:</strong>
            <pre className="log">{info['Error logs']}</pre>
          </div>
        </div>
      ))}
    </div>
  );
};

export default PM2Dashboard;
