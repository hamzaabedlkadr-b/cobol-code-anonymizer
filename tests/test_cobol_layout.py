"""Unittest regressions for lightweight layout hints and full scan coverage."""

from __future__ import annotations

import unittest

from cobol_code_anonymizer.cobol_layout import (
    CODE,
    COMMENT,
    FIXED_COBOL,
    FREE_COBOL,
    JCL,
    LINE_ENDING,
    LITERAL,
    PLAIN_TEXT,
    TEXT,
    classify_source,
)
from cobol_code_anonymizer.source_reader import decode_source_bytes


def layout_for(text: str, path: str):
    source = decode_source_bytes(text.encode("utf-8"), source_path=path)
    return classify_source(source, source_path=path)


class LayoutTests(unittest.TestCase):
    def region_texts(self, layout, kind: str) -> list[str]:
        return [layout.text_for(region) for region in layout.regions_of_kind(kind)]

    def assert_complete_exact_partition(self, layout) -> None:
        self.assertEqual(layout.detector_text, layout.source.text)
        self.assertEqual(
            "".join(layout.text_for(region) for region in layout.regions),
            layout.source.text,
        )
        for region in layout.regions:
            self.assertEqual(
                layout.source.original_byte_span(
                    region.decoded_start, region.decoded_end
                ),
                (region.original_start, region.original_end),
            )

    def test_only_the_five_hint_region_kinds_are_produced(self):
        layouts = [
            layout_for("000100*comment\n000200 DISPLAY 'value'.\n", "A.CBL"),
            layout_for("*> comment\nDISPLAY 'value'.\n", "B.CBL"),
            layout_for("//* comment\n//JOB JOB (A),'value'\n", "C.JCL"),
            layout_for("ordinary text\n", "D.TXT"),
        ]

        kinds = {region.kind for layout in layouts for region in layout.regions}
        self.assertEqual(kinds, {COMMENT, LITERAL, CODE, TEXT, LINE_ENDING})
        for index, layout in enumerate(layouts):
            with self.subTest(layout=index):
                self.assert_complete_exact_partition(layout)

    def test_fixed_continuation_quote_reopens_the_carried_literal(self):
        text = (
            "000100     MOVE 'ANNA VER\n"
            "000200-    'DI' TO WS-NOTE.\n"
            "000300     DISPLAY 'NEXT'.\n"
        )

        layout = layout_for(text, "SAMPLE.CBL")
        literals = layout.regions_of_kind(LITERAL)

        self.assertEqual(
            [layout.text_for(region) for region in literals],
            ["'ANNA VER", "'DI'", "'NEXT'"],
        )
        self.assertEqual([region.continued for region in literals], [False, True, False])
        self.assert_complete_exact_partition(layout)

    def test_non_continuation_fixed_line_resets_unclosed_quote_state(self):
        layout = layout_for(
            "000100     DISPLAY 'BROKEN\n000200     DISPLAY 'VALID'.\n",
            "RESET.CBL",
        )

        self.assertEqual(self.region_texts(layout, LITERAL), ["'BROKEN", "'VALID'"])
        self.assertFalse(any(region.continued for region in layout.regions))

    def test_inline_comment_marker_inside_literal_is_not_a_comment(self):
        layout = layout_for(
            "000100     DISPLAY 'text *> literal' *> actual comment\n",
            "COMMENTS.CBL",
        )

        self.assertEqual(self.region_texts(layout, LITERAL), ["'text *> literal'"])
        self.assertEqual(self.region_texts(layout, COMMENT), ["*> actual comment"])

    def test_non_source_files_are_whole_line_text_without_quote_parsing(self):
        paths = [
            "notes.txt",
            "data.csv",
            "query.sql",
            "records.dat",
            "load.ctl",
            "readme.md",
            "doc.xml",
            "data.json",
            "run.log",
            "README",
        ]
        for path in paths:
            with self.subTest(path=path):
                layout = layout_for("A 'quoted value' remains text\n", path)
                self.assertEqual(layout.format, PLAIN_TEXT)
                self.assertEqual(
                    self.region_texts(layout, TEXT),
                    ["A 'quoted value' remains text"],
                )
                self.assertEqual(self.region_texts(layout, LITERAL), [])
                self.assert_complete_exact_partition(layout)

    def test_comments_and_literals_are_only_hints_in_cobol_and_jcl(self):
        free = layout_for(
            ">>SOURCE FORMAT FREE\nDISPLAY \"inside *> literal\" *> comment\n",
            "FREE.CBL",
        )
        jcl = layout_for("//* comment\n//JOB JOB (A),'operator'\n", "RUN.JCL")

        self.assertEqual(free.format, FREE_COBOL)
        self.assertEqual(self.region_texts(free, LITERAL), ['"inside *> literal"'])
        self.assertEqual(self.region_texts(free, COMMENT), ["*> comment"])
        self.assertEqual(jcl.format, JCL)
        self.assertEqual(self.region_texts(jcl, COMMENT), ["//* comment"])
        self.assertEqual(self.region_texts(jcl, LITERAL), ["'operator'"])

    def test_known_extension_wins_and_extensionless_source_needs_evidence(self):
        self.assertEqual(
            layout_for("// ordinary COBOL content\n", "SOURCE.CBL").format,
            FREE_COBOL,
        )
        self.assertEqual(
            layout_for("// ordinary text content\n", "SOURCE.TXT").format,
            PLAIN_TEXT,
        )
        self.assertEqual(layout_for("//JOB JOB (A),'RUN'\n", "JOBFILE").format, JCL)
        self.assertEqual(
            layout_for("IDENTIFICATION DIVISION.\n", "PROGRAM").format,
            FREE_COBOL,
        )
        self.assertEqual(
            layout_for("an ordinary extensionless file\n", "README").format,
            PLAIN_TEXT,
        )

    def test_tabs_never_make_layout_a_scan_gate(self):
        layout = layout_for("000100 \tDISPLAY 'VALUE'.\n", "TABS.CBL")

        self.assertEqual(layout.format, FIXED_COBOL)
        self.assertEqual(layout.detector_text, "000100 \tDISPLAY 'VALUE'.\n")
        self.assert_complete_exact_partition(layout)

    def test_utf8_region_byte_offsets_remain_exact(self):
        layout = layout_for(">>SOURCE FORMAT FREE\nDISPLAY 'città'.\n", "UTF8.CBL")
        literal = layout.regions_of_kind(LITERAL)[0]

        self.assertEqual(layout.text_for(literal), "'città'")
        self.assertEqual(
            literal.original_end - literal.original_start,
            len("'città'".encode("utf-8")),
        )


if __name__ == "__main__":
    unittest.main()
