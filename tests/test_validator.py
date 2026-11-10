import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validator import validate_functional_errors


class FunctionalErrorValidationTests(unittest.TestCase):
    def test_detects_incompatible_literal_and_assigned_name_operations(self):
        examples = (
            'result = 2 + "text"',
            'count = 2\nresult = count + "text"',
        )
        for code in examples:
            with self.subTest(code=code):
                error = validate_functional_errors(code)
                self.assertIsNotNone(error)
                self.assertEqual(error["error"], "TypeError")
                self.assertEqual(error["node"], "BinOp")

    def test_detects_integer_operator_and_augmented_assignment_errors(self):
        examples = (
            'result = 2 << "text"',
            'value = 1\nvalue += "text"',
            'result = "text" - 1',
            'result = 1j // 2',
        )
        for code in examples:
            with self.subTest(code=code):
                self.assertEqual(
                    validate_functional_errors(code)["error"], "TypeError"
                )

    def test_detects_zero_division(self):
        error = validate_functional_errors("result = 10 / 0")
        self.assertEqual(error["error"], "ZeroDivisionError")
        self.assertEqual(error["line"], 1)

    def test_tracks_types_inside_function_without_leaking_them(self):
        code = (
            "value = 1\n"
            "def calculate():\n"
            '    local = "text"\n'
            "    return local + 1\n"
        )
        error = validate_functional_errors(code)
        self.assertEqual(error["error"], "TypeError")
        self.assertEqual(error["line"], 4)

    def test_uses_annotations_as_static_type_evidence(self):
        examples = (
            "def calculate(value: int):\n    return value + 'text'",
            "value: int\nresult = value + 'text'",
        )
        for code in examples:
            with self.subTest(code=code):
                self.assertEqual(
                    validate_functional_errors(code)["error"], "TypeError"
                )

    def test_branch_assignments_do_not_leak_into_unconditional_code(self):
        code = (
            "value = 1\n"
            "if condition:\n"
            '    value = "text"\n'
            "result = value + 1\n"
        )
        self.assertIsNone(validate_functional_errors(code))

    def test_accepts_valid_numeric_and_sequence_operations(self):
        code = (
            "number = 2 + 3.5\n"
            'text = "a" + "b"\n'
            'repeated = "a" * 3\n'
            "bits = 3 << 1\n"
        )
        self.assertIsNone(validate_functional_errors(code))

    def test_reports_syntax_errors(self):
        error = validate_functional_errors("if True print('broken')")
        self.assertEqual(error["error"], "SyntaxError")


if __name__ == "__main__":
    unittest.main()
