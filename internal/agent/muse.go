package agent

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gofrs/flock"
	"github.com/google/uuid"
	"go.kenn.io/kit/atomicfile"
)

// MuseAgent runs Meta's Muse Code CLI in headless mode.
type MuseAgent struct {
	Command   string
	Model     string
	Reasoning ReasoningLevel
	Agentic   bool
	SessionID string
}

func NewMuseAgent(command string) *MuseAgent {
	if command == "" {
		command = "muse"
	}
	return &MuseAgent{Command: command, Reasoning: ReasoningStandard}
}

func (a *MuseAgent) clone(opts ...agentCloneOption) *MuseAgent {
	cfg := newAgentCloneConfig(a.Command, a.Model, a.Reasoning, a.Agentic, a.SessionID, opts...)
	return &MuseAgent{Command: cfg.Command, Model: cfg.Model, Reasoning: cfg.Reasoning, Agentic: cfg.Agentic, SessionID: cfg.SessionID}
}

func (a *MuseAgent) WithReasoning(level ReasoningLevel) Agent {
	return a.clone(withClonedReasoning(level))
}

func (a *MuseAgent) WithAgentic(agentic bool) Agent {
	return a.clone(withClonedAgentic(agentic))
}

func (a *MuseAgent) WithModel(model string) Agent {
	if model == "" {
		return a
	}
	return a.clone(withClonedModel(model))
}

func (a *MuseAgent) WithSessionID(sessionID string) Agent {
	return a.clone(withClonedSessionID(sessionID))
}

func (a *MuseAgent) Name() string        { return "muse" }
func (a *MuseAgent) CommandName() string { return a.Command }

func (a *MuseAgent) buildArgs(agentic bool, promptFile string) []string {
	return a.buildArgsWithSession(agentic, promptFile, a.SessionID)
}

func (a *MuseAgent) buildArgsWithSession(agentic bool, promptFile, sessionID string) []string {
	args := []string{"exec"}
	if sessionID != "" {
		args = append(args, "--session-id", sessionID)
	} else {
		args = append(args, "--no-session-log")
	}
	args = append(args, "--contributor-mode", "off")
	if a.Model != "" {
		args = append(args, "--model", a.Model)
	}
	if effort := reasoningEffort(a.Reasoning); effort != "" {
		args = append(args, "--reasoning-effort", effort)
	}
	if agentic {
		args = append(args, "--yolo")
	} else {
		args = append(args, "--no-foreign-personal-context", "-c", `run.toolset=["read_file","search"]`, "-c", `hooks={}`, "-c", "managed_hooks_path=null")
	}
	return append(args, "--prompt-file", promptFile)
}

func (a *MuseAgent) CommandLine() string {
	return a.Command + " " + strings.Join(a.buildArgs(a.Agentic || AllowUnsafeAgents(), "<prompt-file>"), " ")
}

func (a *MuseAgent) PlanningCommandLine() string {
	return a.Command + " " + strings.Join(a.buildArgsWithSession(false, "<prompt-file>", ""), " ")
}

func (a *MuseAgent) Review(ctx context.Context, repoPath, commitSHA, prompt string, output io.Writer) (string, error) {
	// Muse requires a real file for prompts that exceed argv limits.
	agentic := effectiveAgentic(ctx, a.Agentic)
	promptPath, env, cleanup, err := prepareMuseRunWithSession(prompt, agentic, a.SessionID)
	if err != nil {
		return "", err
	}
	defer cleanup()
	sessionID := a.SessionID
	if !planningReadOnly(ctx) && sessionID == "" {
		sessionID = uuid.NewString()
	}
	if reporter, ok := output.(interface{ ReportSessionID(string) }); ok && sessionID != "" {
		reporter.ReportSessionID(sessionID)
	}
	result, err := runStreamingCLI(ctx, streamingCLISpec{
		Name: "muse", Command: a.Command,
		Args: a.buildArgsWithSession(agentic, promptPath, sessionID),
		Dir:  repoPath, Env: env, Output: output, StreamStderr: true, DrainStdout: true,
		Parse: parsePlainTextOutput,
	})
	if err != nil {
		return "", err
	}
	if result.WaitErr != nil {
		return "", formatStreamingCLIWaitError("muse", result, result.Stderr)
	}
	if result.ParseErr != nil {
		return result.Result, result.ParseErr
	}
	if result.Result == "" {
		return "No review output generated", nil
	}
	return result.Result, nil
}

// prepareMuseRun writes the prompt file. Non-agentic runs also get a private
// home holding only the Muse login and settings, plus empty data, cache, and
// temp directories, to exclude saved workspace trust and installed user
// plugins. The private temp directory keeps Muse's runtime files out of the
// system temp directory that holds planning worktrees.
func prepareMuseRun(prompt string, agentic bool) (string, []string, func(), error) {
	return prepareMuseRunWithSession(prompt, agentic, "")
}

func prepareMuseRunWithSession(prompt string, agentic bool, resumeSessionID string) (string, []string, func(), error) {
	promptPath, cleanupPrompt, err := createPromptFile("muse", prompt)
	if err != nil || agentic {
		return promptPath, nil, cleanupPrompt, err
	}
	src := filepath.Join(os.Getenv("XDG_CONFIG_HOME"), "muse")
	home := ""
	if os.Getenv("XDG_CONFIG_HOME") == "" || os.Getenv("XDG_DATA_HOME") == "" {
		var homeErr error
		home, homeErr = os.UserHomeDir()
		if homeErr != nil && os.Getenv("XDG_CONFIG_HOME") == "" {
			cleanupPrompt()
			return "", nil, nil, fmt.Errorf("resolve muse config home: %w", homeErr)
		}
	}
	if os.Getenv("XDG_CONFIG_HOME") == "" {
		src = filepath.Join(home, ".config", "muse")
	}
	userDataMuseDir := ""
	if xdgData := os.Getenv("XDG_DATA_HOME"); xdgData != "" {
		userDataMuseDir = filepath.Join(xdgData, "muse")
	} else if home != "" {
		userDataMuseDir = filepath.Join(home, ".local", "share", "muse")
	}
	dir, err := os.MkdirTemp("", "roborev-muse-config-*")
	if err != nil {
		cleanupPrompt()
		return "", nil, nil, fmt.Errorf("create muse config dir: %w", err)
	}
	cleanupAuth := func() {}
	cleanupSessions := func() {}
	cleanup := func() { cleanupAuth(); cleanupSessions(); cleanupPrompt(); _ = os.RemoveAll(dir) }
	configDir := filepath.Join(dir, ".config")
	dataDir := filepath.Join(dir, "data")
	cacheDir := filepath.Join(dir, "cache")
	tmpDir := filepath.Join(dir, "tmp")
	for _, path := range []string{configDir, dataDir, cacheDir, tmpDir} {
		if err := os.Mkdir(path, 0o700); err != nil {
			cleanup()
			return "", nil, nil, fmt.Errorf("create muse private directory: %w", err)
		}
	}
	err = os.Mkdir(filepath.Join(configDir, "muse"), 0o700)
	for _, name := range []string{"auth.json", "settings.json"} {
		data, readErr := os.ReadFile(filepath.Join(src, name))
		if err == nil && readErr == nil && name == "settings.json" {
			data, err = museReadOnlySettings(data)
		}
		if err == nil && readErr == nil {
			err = os.WriteFile(filepath.Join(configDir, "muse", name), data, 0o600)
			if err == nil && name == "auth.json" {
				cleanupAuth = func() {
					saveMuseAuth(filepath.Join(configDir, "muse", name), filepath.Join(src, name), data)
				}
			}
		} else if err == nil && !errors.Is(readErr, fs.ErrNotExist) {
			err = readErr
		}
	}
	if err != nil {
		cleanup()
		return "", nil, nil, fmt.Errorf("copy muse config: %w", err)
	}
	if userDataMuseDir != "" {
		userSessionsDir := filepath.Join(userDataMuseDir, "sessions")
		privateSessionsDir := filepath.Join(dataDir, "muse", "sessions")
		if resumeSessionID != "" {
			if info, statErr := os.Stat(userSessionsDir); statErr == nil && info.IsDir() {
				_ = copyMuseSessionByID(userSessionsDir, privateSessionsDir, resumeSessionID)
			}
		}
		cleanupSessions = func() {
			if info, statErr := os.Stat(privateSessionsDir); statErr == nil && info.IsDir() {
				_ = copyMuseSessionsTree(privateSessionsDir, userSessionsDir)
			}
		}
	}
	return promptPath, append(os.Environ(), "HOME="+dir, "USERPROFILE="+dir, "XDG_CONFIG_HOME="+configDir, "XDG_DATA_HOME="+dataDir, "XDG_CACHE_HOME="+cacheDir, "TMPDIR="+tmpDir, "TMP="+tmpDir, "TEMP="+tmpDir), cleanup, nil
}

func copyMuseSessionByID(srcSessionsDir, dstSessionsDir, sessionID string) error {
	return filepath.WalkDir(srcSessionsDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if !d.IsDir() || d.Name() != sessionID {
			return nil
		}
		rel, err := filepath.Rel(srcSessionsDir, path)
		if err != nil {
			return err
		}
		return copyMuseSessionsTree(path, filepath.Join(dstSessionsDir, rel))
	})
}

func copyMuseSessionsTree(srcDir, dstDir string) error {
	return filepath.WalkDir(srcDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dstDir, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o700)
		}
		if !d.Type().IsRegular() || filepath.Base(path) == ".session.lock" {
			return nil
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return atomicfile.WriteFile(target, data, atomicfile.WithPrivate(), atomicfile.WithFollowLink())
	})
}

// writeBackMuseAuth saves a login Muse renewed in the private home, unless
// another Muse login or run has changed the user's file since it was copied.
func writeBackMuseAuth(privatePath, userPath string, original []byte) error {
	updated, err := os.ReadFile(privatePath)
	if err != nil || bytes.Equal(updated, original) {
		return err
	}
	return replaceMuseAuthIfUnchanged(userPath, original, updated)
}

// saveMuseAuth reports a renewed login it could not save, without failing the run.
func saveMuseAuth(privatePath, userPath string, original []byte) {
	if err := writeBackMuseAuth(privatePath, userPath, original); err != nil {
		log.Printf("muse: could not save renewed login to %s: %v", userPath, err)
	}
}

// replaceMuseAuthIfUnchanged holds Muse's auth lock from the comparison
// through the replacement, so a concurrent Muse login is never overwritten.
func replaceMuseAuthIfUnchanged(userPath string, original, updated []byte) error {
	unlock, err := lockMuseAuth(userPath)
	if err != nil {
		return err
	}
	defer unlock()
	current, err := os.ReadFile(userPath)
	if err != nil || !bytes.Equal(current, original) {
		return err
	}
	return atomicfile.WriteFile(userPath, updated, atomicfile.WithPrivate(), atomicfile.WithFollowLink())
}

// lockMuseAuth takes the lock Muse holds while it writes auth.json. It does
// not wait for another holder, so a renewal that meets a concurrent Muse login
// or run is reported and dropped instead of raced.
func lockMuseAuth(userPath string) (func(), error) {
	lockPath := filepath.Join(filepath.Dir(userPath), ".auth.json.lock")
	lock := flock.New(lockPath, flock.SetPermissions(0o600))
	locked, err := lock.TryLock()
	if err == nil && !locked {
		err = errors.New("held by another process")
	}
	if err != nil {
		_ = lock.Close()
		return nil, fmt.Errorf("lock %s: %w", lockPath, err)
	}
	return func() { _ = lock.Close() }, nil
}

// museReadOnlySettings preserves user settings except MCP server declarations.
func museReadOnlySettings(data []byte) ([]byte, error) {
	var settings map[string]jsontext.Value
	if err := json.Unmarshal(data, &settings); err != nil {
		return nil, fmt.Errorf("read muse settings: %w", err)
	}
	var version int
	if err := json.Unmarshal(settings["schema_version"], &version); err != nil {
		return nil, fmt.Errorf("muse settings require schema_version 1: %w", err)
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported muse settings schema_version %d; expected 1", version)
	}
	delete(settings, "mcpServers")
	delete(settings, "mcp_servers")
	return json.Marshal(settings)
}

func init() { Register(NewMuseAgent("")) }
