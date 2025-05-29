import { useEffect, useState } from "react";

const fetchData = async () => {
  const res = await fetch("http://localhost:8080");
  if (!res.ok) throw new Error("Failed to fetch");
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
            <div className="app-line">
              <div className="left-side">
                <strong className="app-name">{name}</strong>
                <span className="args">: {info.Args.join(" ")}</span>
              </div>

              <div className="right-side">
                <span
                  className={`status ${
                    info.State === "online" ? "online" : "offline"
                  }`}
                >
                  {info.State}
                </span>
                <button
                  className="arrow-btn"
                  onClick={() => toggleExpand(name)}
                >
                  {expanded[name] ? "▾" : "▸"}
                </button>
              </div>
            </div>
          </div>

          {expanded[name] && (
            <div className="card-section">
              <pre className="log">{info["Error logs"]}</pre>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default PM2Dashboard;
