import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import END, MAX_TOOL_ROUNDS, after_tools, count_tools, finalize_report, should_continue
from tools import read_failure, report_path, write_report


class FinishWithoutReport(unittest.TestCase):
    def test_successful_read_then_text_reply_routes_to_finalize(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                HumanMessage(content="Research this topic and write a report: quantum computing"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_url",
                            "args": {"url": "https://example.com"},
                            "id": "1",
                        }
                    ],
                ),
                ToolMessage(
                    content="Example Domain\nThis domain is for use in illustrative examples.",
                    tool_call_id="1",
                    name="read_url",
                ),
                AIMessage(content="# Research Report\nFindings from the source."),
            ],
        }
        self.assertEqual(should_continue(state), "finalize")

    def test_no_successful_read_ends_without_report(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                HumanMessage(content="Research this topic and write a report: quantum computing"),
                AIMessage(content="SEARCH_EMPTY on every query. I cannot find sources."),
            ],
        }
        self.assertEqual(should_continue(state), END)

    def test_finalize_writes_last_model_text(self):
        reports = Path("reports").resolve()
        before = set(reports.glob("quantum_computing_*.md"))
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(
                    content="Example Domain\nThis domain is for use in illustrative examples.",
                    tool_call_id="1",
                    name="read_url",
                ),
                AIMessage(content="# Research Report\nFindings from the source."),
            ],
        }
        result = finalize_report(state)
        saved = Path(
            result["messages"][0].content.removeprefix("Report saved to ")
        ).resolve()
        self.addCleanup(saved.unlink)
        self.assertEqual(saved.parent, reports)
        body = saved.read_text()
        self.assertIn("Findings from the source.", body)
        self.assertIn("Example Domain", body)
        self.assertTrue(saved not in before)

    def test_page_text_with_failure_prefix_still_finalizes(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(
                    content="Failed to read URL: this line is the article.",
                    tool_call_id="1",
                    name="read_url",
                ),
                AIMessage(content="Findings."),
            ],
        }
        self.assertEqual(should_continue(state), "finalize")

    def test_marked_read_failure_ends_without_report(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(
                    content=read_failure("timeout"),
                    tool_call_id="1",
                    name="read_url",
                ),
                AIMessage(content="No sources."),
            ],
        }
        self.assertEqual(should_continue(state), END)

    def test_tool_cap_finalizes_when_a_page_was_read(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(content="Example Domain", tool_call_id="1", name="read_url"),
            ],
            "tool_rounds": MAX_TOOL_ROUNDS,
        }
        self.assertEqual(after_tools(state), "finalize")

    def test_tool_cap_ends_when_nothing_was_read(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(content=read_failure("timeout"), tool_call_id="1", name="read_url"),
            ],
            "tool_rounds": MAX_TOOL_ROUNDS,
        }
        self.assertEqual(after_tools(state), END)

    def test_count_tools_increments_by_batch_size(self):
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "a"}, "id": "1"},
                        {"name": "search_web", "args": {"query": "b"}, "id": "2"},
                    ],
                ),
                ToolMessage(content="one", tool_call_id="1", name="search_web"),
                ToolMessage(content="two", tool_call_id="2", name="search_web"),
            ],
            "topic": "t",
            "tool_rounds": 2,
        }
        self.assertEqual(count_tools(state), {"tool_rounds": 4})

    def test_next_batch_that_would_pass_the_cap_finalizes(self):
        state = {
            "topic": "quantum computing",
            "tool_rounds": MAX_TOOL_ROUNDS - 1,
            "messages": [
                ToolMessage(content="Example Domain", tool_call_id="1", name="read_url"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_web", "args": {"query": "a"}, "id": "2"},
                        {"name": "search_web", "args": {"query": "b"}, "id": "3"},
                    ],
                ),
            ],
        }
        self.assertEqual(should_continue(state), "finalize")

    def test_finalize_after_a_tool_result_uses_the_page(self):
        reports = Path("reports").resolve()
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(content="Example Domain page body", tool_call_id="1", name="read_url"),
                ToolMessage(content="SEARCH_EMPTY: No results found.", tool_call_id="2", name="search_web"),
            ],
        }
        result = finalize_report(state)
        saved = Path(result["messages"][0].content.removeprefix("Report saved to ")).resolve()
        self.addCleanup(saved.unlink)
        body = saved.read_text()
        self.assertIn("Example Domain page body", body)
        self.assertNotIn("SEARCH_EMPTY", body)
        self.assertEqual(saved.parent, reports)

    def test_finalize_keeps_prose_that_came_with_the_tool_call(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                AIMessage(
                    content="Qubits stay coherent longer in this setup.",
                    tool_calls=[{"name": "read_url", "args": {"url": "https://example.com"}, "id": "1"}],
                ),
                ToolMessage(content="Example Domain page body", tool_call_id="1", name="read_url"),
            ],
        }
        result = finalize_report(state)
        saved = Path(result["messages"][0].content.removeprefix("Report saved to ")).resolve()
        self.addCleanup(saved.unlink)
        body = saved.read_text()
        self.assertIn("Qubits stay coherent longer", body)
        self.assertIn("Example Domain page body", body)

    def test_successful_write_ends_after_tools(self):
        state = {
            "topic": "quantum computing",
            "messages": [
                ToolMessage(
                    content="Example Domain",
                    tool_call_id="1",
                    name="read_url",
                ),
                ToolMessage(
                    content="Report saved to reports/quantum_computing_20261004_0000.md",
                    tool_call_id="2",
                    name="write_report",
                ),
            ],
        }
        self.assertEqual(after_tools(state), END)


class ReportPathStaysInReports(unittest.TestCase):
    def test_dotdot_topic_writes_only_under_reports(self):
        reports = Path("reports").resolve()
        cwd = Path(".").resolve()
        before_cwd = {p.name for p in cwd.iterdir()}
        result = write_report.invoke(
            {"topic": "../escaped_probe", "content": "owned"}
        )
        self.assertTrue(result.startswith("Report saved to "))
        saved = Path(result.removeprefix("Report saved to ")).resolve()
        self.assertEqual(saved.parent, reports)
        self.assertTrue(saved.is_file())
        after_cwd = {p.name for p in cwd.iterdir()}
        self.assertEqual(after_cwd - before_cwd, set())
        saved.unlink()

    def test_same_stamp_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            reports = Path(raw)
            when = datetime(2026, 10, 4, 1, 8, 0)
            first = report_path("quantum computing", when=when, reports_dir=reports)
            first.write_text("first")
            second = report_path("quantum computing", when=when, reports_dir=reports)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(), "first")
            self.assertEqual(second.parent, reports.resolve())

    def test_symlinked_reports_dir_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            outside = base / "outside"
            outside.mkdir()
            link = base / "reports"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaises(ValueError):
                report_path("topic", reports_dir=link)
            self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
