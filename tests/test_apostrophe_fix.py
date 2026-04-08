"""
Test for the apostrophe/quote escaping fix.

Verifies that html.escape(quote=False) prevents &#x27; from appearing
in HTML, VTT, and TXT output.

TODO: change these test functions to use the actual functions from noscribe
after refactoring the code.
"""

import html

import AdvancedHTMLParser


def _build_doc_with_text(text: str) -> AdvancedHTMLParser.AdvancedHTMLParser:
    """Simulate how noScribe builds the HTML DOM with transcript text."""
    d = AdvancedHTMLParser.AdvancedHTMLParser()
    d.parseStr('<html><body></body></html>')

    p = d.createElement('p')
    d.body.appendChild(p)

    # This mirrors line 2859 + 2922-2923 of noScribe.py (after the fix)
    seg_html = html.escape(text, quote=False)
    a_html = f'<a name="ts_0_1000_S01" >{seg_html}</a>'
    a = d.createElementFromHTML(a_html)
    p.appendChild(a)

    return d


class TestApostropheFix:
    """Tests that apostrophes are correctly preserved in all output formats."""

    def test_html_output_no_entity(self):
        """HTML output should contain literal apostrophe, not &#x27;"""
        d = _build_doc_with_text("Romy's story")
        html_out = d.asHTML()
        assert "Romy's" in html_out, f"Expected literal apostrophe in HTML output, got: {html_out}"
        assert "&#x27;" not in html_out, f"Found &#x27; entity in HTML output: {html_out}"
    def test_html_still_escapes_dangerous_chars(self):
        """Ensure that <, >, & are still properly escaped."""
        d = _build_doc_with_text("a < b & c > d")
        html_out = d.asHTML()
        assert "&lt;" in html_out, "< should be escaped to &lt;"
        assert "&amp;" in html_out, "& should be escaped to &amp;"
        assert "&gt;" in html_out, "> should be escaped to &gt;"

    def test_double_quotes_not_escaped(self):
        """Double quotes should also not be escaped in text content."""
        d = _build_doc_with_text('She said "hello"')
        html_out = d.asHTML()
        assert '&quot;' not in html_out, f"Found &quot; entity in HTML output: {html_out}"
        assert '"hello"' in html_out, f"Expected literal double quotes in HTML output: {html_out}"

    def test_various_apostrophe_cases(self):
        """Test various common apostrophe patterns in transcripts."""
        test_cases = [
            "It's a test",
            "Don't worry",
            "I'm fine",
            "That's Romy's idea",
            "The '90s were great",
        ]
        for text in test_cases:
            d = _build_doc_with_text(text)
            html_out = d.asHTML()
            assert "&#x27;" not in html_out, f"Found &#x27; for input '{text}': {html_out}"
            assert "&#39;" not in html_out, f"Found &#39; for input '{text}': {html_out}"
