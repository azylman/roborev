package daemon

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.kenn.io/roborev/internal/config"
	gitpkg "go.kenn.io/roborev/internal/git"
	"go.kenn.io/roborev/internal/storage"
	"go.kenn.io/roborev/internal/testutil"
)

func TestFindCompatibleReusableSessionUsesResolvedAgentCLI(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake Gemini command uses a POSIX shell")
	}
	server, db, tmpDir := newTestServer(t)
	repoDir := filepath.Join(tmpDir, "testrepo")
	testRepo := testutil.InitTestGitRepo(t, repoDir)
	testRepo.CheckoutNewBranch("feature/session")

	reuseSessions := true
	cfg := server.configWatcher.Config()
	cfg.ReuseReviewSession = &reuseSessions
	cfg.ReuseReviewSessionLookback = 1
	geminiCommand := filepath.Join(tmpDir, "gemini")
	require.NoError(t, os.WriteFile(geminiCommand, []byte("#!/bin/sh\nexit 0\n"), 0o755))
	cfg.GeminiCmd = geminiCommand

	repoRoot, err := gitpkg.GetMainRepoRoot(repoDir)
	require.NoError(t, err)
	repo, err := db.GetOrCreateRepo(repoRoot, config.ResolveRepoIdentity(repoRoot, nil))
	require.NoError(t, err)
	sha := testRepo.HeadSHA()
	commit, err := db.GetOrCreateCommit(repo.ID, sha, "Author", "Subject", time.Now())
	require.NoError(t, err)

	seedCandidate := func(sessionID, sessionCLI string, finishedAt time.Time) {
		t.Helper()
		job, err := db.EnqueueJob(storage.EnqueueOpts{
			RepoID: repo.ID, CommitID: commit.ID, GitRef: sha, Branch: "feature/session",
			Agent: "gemini", Model: "gemini-3.1-pro-preview", Reasoning: "thorough",
			ReviewType: config.ReviewTypeDefault,
		})
		require.NoError(t, err)
		claimed, err := db.ClaimJob("session-worker")
		require.NoError(t, err)
		require.Equal(t, job.ID, claimed.ID)
		require.NoError(t, testutil.CompleteReviewFixture(db, job.ID, "gemini", "prompt", "No issues found."))
		_, err = db.ExecContext(t.Context(), `
			UPDATE review_jobs
			SET session_id = ?, session_cli = ?, finished_at = ?
			WHERE id = ?`,
			sessionID, sessionCLI, finishedAt.Format(time.RFC3339Nano), job.ID,
		)
		require.NoError(t, err)
	}
	now := time.Now()
	seedCandidate("older-gemini-session", "gemini", now.Add(-2*time.Minute))
	seedCandidate("newer-agy-session", "agy", now.Add(-time.Minute))

	opts := storage.EnqueueOpts{
		RepoID: repo.ID, GitRef: sha, Branch: "feature/session",
		Agent: "gemini", Model: "gemini-3.1-pro-preview", Reasoning: "thorough",
		ReviewType: config.ReviewTypeDefault,
	}
	sessionID, _, _ := findCompatibleReusableSession(
		t.Context(), db, repoRoot, sha, opts, nil, nil, cfg, nil, 0,
	)
	assert.Equal(t, "older-gemini-session", sessionID)
}

func TestWorkerResolvesReusableSessionAfterFailover(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake agent command uses a POSIX shell")
	}
	server, db, tmpDir := newTestServer(t)
	repoDir := filepath.Join(tmpDir, "testrepo")
	testRepo := testutil.InitTestGitRepo(t, repoDir)
	testRepo.CheckoutNewBranch("feature/failover-reuse")

	reuseSessions := true
	cfg := server.configWatcher.Config()
	cfg.ReuseReviewSession = &reuseSessions
	museCommand := filepath.Join(tmpDir, "muse")
	argsLog := filepath.Join(tmpDir, "muse-args.txt")
	doc := string(testutil.ReviewFixtureJSON("No issues found."))
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '" + argsLog + "'\ncat <<'EOF'\n" + doc + "\nEOF\n"
	require.NoError(t, os.WriteFile(museCommand, []byte(script), 0o755))
	cfg.MuseCmd = museCommand

	repoRoot, err := gitpkg.GetMainRepoRoot(repoDir)
	require.NoError(t, err)
	repo, err := db.GetOrCreateRepo(repoRoot, config.ResolveRepoIdentity(repoRoot, nil))
	require.NoError(t, err)
	sha1 := testRepo.HeadSHA()
	commit1, err := db.GetOrCreateCommit(repo.ID, sha1, "Author", "Subject 1", time.Now())
	require.NoError(t, err)

	// Seed Round 1: a panel member job that failed over to muse and completed with session "muse-round1-session".
	job1, err := db.EnqueueJob(storage.EnqueueOpts{
		RepoID: repo.ID, CommitID: commit1.ID, GitRef: sha1, Branch: "feature/failover-reuse",
		Agent: "muse", Model: "muse-spark-1.3-contributor", Reasoning: "thorough",
		ReviewType: config.ReviewTypeDefault,
		PanelRole:  storage.PanelRoleMember, PanelName: "default", PanelMemberName: "default",
		PanelMemberConfigJSON: `{"agent":"claude-code"}`,
	})
	require.NoError(t, err)
	claimed1, err := db.ClaimJob(testWorkerID)
	require.NoError(t, err)
	require.Equal(t, job1.ID, claimed1.ID)
	require.NoError(t, testutil.CompleteReviewFixture(db, job1.ID, "muse", "prompt", "No issues found."))
	_, err = db.ExecContext(t.Context(), `
		UPDATE review_jobs SET session_id = 'muse-round1-session' WHERE id = ?`, job1.ID)
	require.NoError(t, err)

	// Create Round 2 on a successor commit, initially enqueued for claude-code (so session_id is empty),
	// then failed over to muse via FailoverJob.
	sha2 := testRepo.CommitFile("next.txt", "next\n", "Subject 2")
	commit2, err := db.GetOrCreateCommit(repo.ID, sha2, "Author", "Subject 2", time.Now())
	require.NoError(t, err)
	job2, err := db.EnqueueJob(storage.EnqueueOpts{
		RepoID: repo.ID, CommitID: commit2.ID, GitRef: sha2, Branch: "feature/failover-reuse",
		Agent: "claude-code", Model: "sonnet", Reasoning: "thorough",
		ReviewType: config.ReviewTypeDefault,
		PanelRole:  storage.PanelRoleMember, PanelName: "default", PanelMemberName: "default",
		PanelMemberConfigJSON: `{"agent":"claude-code"}`,
	})
	require.NoError(t, err)
	claimed2, err := db.ClaimJob(testWorkerID)
	require.NoError(t, err)
	require.Equal(t, job2.ID, claimed2.ID)
	failedOver, err := db.FailoverJob(job2.ID, testWorkerID, "muse", "muse-spark-1.3-contributor")
	require.NoError(t, err)
	require.True(t, failedOver)

	claimedFailover, err := db.ClaimJob(testWorkerID)
	require.NoError(t, err)
	require.NotNil(t, claimedFailover)
	require.Equal(t, job2.ID, claimedFailover.ID)
	assert.Empty(t, claimedFailover.SessionID)

	server.workerPool.processJob(testWorkerID, claimedFailover)
	stored2, err := db.GetJobByID(job2.ID)
	require.NoError(t, err)
	require.Equal(t, storage.JobStatusDone, stored2.Status, stored2.Error)
	assert.Equal(t, "muse-round1-session", stored2.SessionID)
	require.NotNil(t, stored2.ResumeSourceJobUUID)
	assert.Equal(t, *job1.UUID, *stored2.ResumeSourceJobUUID)

	loggedArgs, err := os.ReadFile(argsLog)
	require.NoError(t, err)
	assert.Contains(t, string(loggedArgs), "--session-id muse-round1-session")
}
