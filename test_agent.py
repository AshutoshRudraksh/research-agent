import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import END, after_tools, finalize_report, should_continue
from tools import write_report


class FinishWithoutReport(unittest.TestCase):
    def test_successful_read_then_text_reply_does_not_end(self):
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
        self.assertIn("Findings from the source.", saved.read_text())
        self.assertTrue(saved not in before)

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


if __name__ == "__main__":
    unittest.main()
