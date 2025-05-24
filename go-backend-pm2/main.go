package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strings"
)

// PM2Process represents a single process info from `pm2 jlist`
type PM2Process struct {
	Name   string `json:"name"`
	Pm2Env struct {
		Status string   `json:"status"`
		Args   []string `json:"args"`
	} `json:"pm2_env"`
}

// ProcessInfo is the format of the JSON response
type ProcessInfo struct {
	State     string   `json:"State"`
	Args      []string `json:"Args"`
	ErrorLogs string   `json:"Error logs"`
}

func getPM2Processes() (map[string]ProcessInfo, error) {
	// log.Println("[DEBUG] Running `pm2 jlist`...")
	cmd := exec.Command("pm2", "jlist")
	output, err := cmd.Output()
	if err != nil {
		// log.Printf("[ERROR] Failed to run pm2 jlist: %v\n", err)
		return nil, err
	}
	// log.Println("[DEBUG] Successfully fetched pm2 jlist")

	var processes []PM2Process
	err = json.Unmarshal(output, &processes)
	if err != nil {
		log.Printf("[ERROR] Failed to unmarshal pm2 jlist JSON: %v\n", err)
		return nil, err
	}
	// log.Printf("[DEBUG] Found %d processes\n", len(processes))

	result := make(map[string]ProcessInfo)

	for _, p := range processes {
		// log.Printf("[DEBUG] Fetching logs for process: %s\n", p.Name)
		logs, err := getErrorLogs(p.Name)
		if err != nil {
			// log.Printf("[WARN] Failed to get error logs for %s: %v\n", p.Name, err)
			logs = "Error fetching logs"
		}

		result[p.Name] = ProcessInfo{
			State:     p.Pm2Env.Status,
			Args:      p.Pm2Env.Args,
			ErrorLogs: logs,
		}
	}

	return result, nil
}

func getErrorLogs(name string) (string, error) {
	cmd := exec.Command("pm2", "logs", name, "--lines", "20", "--err", "--raw", "--nostream")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("log command failed: %w\nOutput:\n%s", err, out.String())
	}

	// Strip ANSI escape sequences (if any remain)
	logOutput := stripANSI(out.String())
	return logOutput, nil
}

func stripANSI(input string) string {
	// Basic ANSI code remover
	replacer := strings.NewReplacer(
		"\x1b[0m", "",
		"\x1b[31m", "",
		"\x1b[32m", "",
		"\x1b[33m", "",
		"\x1b[34m", "",
		"\x1b[35m", "",
		"\x1b[36m", "",
		"\x1b[1m", "",
	)
	return replacer.Replace(input)
}

func handler(w http.ResponseWriter, r *http.Request) {
	log.Println("[DEBUG] HTTP request received")
	processes, err := getPM2Processes()
	if err != nil {
		log.Printf("[ERROR] Handler failed: %v\n", err)
		http.Error(w, "Failed to get PM2 processes: "+err.Error(), http.StatusInternalServerError)
		return
	}
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(processes)
	if err != nil {
		log.Printf("[ERROR] Failed to encode JSON response: %v\n", err)
	}
}

func main() {
	http.HandleFunc("/", handler)
	log.Println("[INFO] Server starting on http://localhost:8080/")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatalf("[FATAL] Server failed to start: %v\n", err)
	}
}
