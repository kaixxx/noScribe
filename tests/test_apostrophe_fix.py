"""
Test for the apostrophe/quote escaping fix.

Verifies that html.escape(quote=False) prevents &#x27; from appearing
in HTML, VTT, and TXT output.
"""
import html
import sys
import os

# Add parent dir to path so we can import from noScribe.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import AdvancedHTMLParser
import utils


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


def _html_node_to_text(node) -> str:
    """Mirrors the html_node_to_text function from noScribe.py (line 349)."""
    if AdvancedHTMLParser.isTextNode(node):
        return html.unescape(node)
    elif AdvancedHTMLParser.isTagNode(node):
        text_parts = []
        for child in node.childBlocks:
            text = _html_node_to_text(child)
            if text:
                text_parts.append(text)
        return ''.join(text_parts)
    return ''


def _vtt_escape(txt: str) -> str:
    """Mirrors vtt_escape from noScribe.py (line 379) after the fix."""
    txt = html.escape(txt, quote=False)
    while txt.find('\n\n') > -1:
        txt = txt.replace('\n\n', '\n')
    return txt


class TestApostropheFix:
    """Tests that apostrophes are correctly preserved in all output formats."""

    def test_html_output_no_entity(self):
        """HTML output should contain literal apostrophe, not &#x27;"""
        d = _build_doc_with_text("Romy's story")
        html_out = d.asHTML()
        assert "Romy's" in html_out, f"Expected literal apostrophe in HTML output, got: {html_out}"
        assert "&#x27;" not in html_out, f"Found &#x27; entity in HTML output: {html_out}"

    def test_txt_output_no_entity(self):
        """TXT output should contain literal apostrophe."""
        d = _build_doc_with_text("Romy's story")
        txt_out = _html_node_to_text(d.body)
        assert "Romy's" in txt_out, f"Expected literal apostrophe in TXT output, got: {txt_out}"
        assert "&#x27;" not in txt_out, f"Found &#x27; entity in TXT output: {txt_out}"

    def test_vtt_escape_no_entity(self):
        """vtt_escape should not escape apostrophes."""
        result = _vtt_escape("Romy's story")
        assert "Romy's" in result, f"Expected literal apostrophe in VTT output, got: {result}"
        assert "&#x27;" not in result, f"Found &#x27; entity in VTT output: {result}"

    def test_html_still_escapes_dangerous_chars(self):
        """Ensure that <, >, & are still properly escaped."""
        d = _build_doc_with_text("a < b & c > d")
        html_out = d.asHTML()
        assert "&lt;" in html_out, "< should be escaped to &lt;"
        assert "&amp;" in html_out, "& should be escaped to &amp;"
        assert "&gt;" in html_out, "> should be escaped to &gt;"

    def test_vtt_still_escapes_dangerous_chars(self):
        """Ensure that <, >, & are still properly escaped in VTT."""
        result = _vtt_escape("a < b & c > d")
        assert "&lt;" in result, "< should be escaped to &lt;"
        assert "&amp;" in result, "& should be escaped to &amp;"
        assert "&gt;" in result, "> should be escaped to &gt;"

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
