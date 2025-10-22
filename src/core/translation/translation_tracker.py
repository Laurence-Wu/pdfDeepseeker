"""
Translation Issue Tracker
Records and manages translation confidence and issues during pipeline execution.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from pathlib import Path


class IssueType(Enum):
    """Types of translation issues"""
    LOW_CONFIDENCE = "low_confidence"
    NO_TRANSLATION = "no_translation"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TECHNICAL_TERM = "technical_term"
    PARTIAL_TRANSLATION = "partial_translation"


class IssueSeverity(Enum):
    """Severity levels for issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TranslationIssue:
    """Represents a single translation issue"""
    issue_type: IssueType
    severity: IssueSeverity
    source_text: str
    translated_text: Optional[str]
    confidence: float
    location: str  # e.g., "page 3, block 5" or "table 0, row 2, cell 1"
    context: Dict[str, Any] = field(default_factory=dict)
    attempted_strategies: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "confidence": self.confidence,
            "location": self.location,
            "context": self.context,
            "attempted_strategies": self.attempted_strategies,
            "error_message": self.error_message
        }


class TranslationTracker:
    """Tracks translation quality and issues throughout the pipeline"""

    def __init__(self):
        self.issues: List[TranslationIssue] = []
        self.total_translations = 0
        self.successful_translations = 0
        self.failed_translations = 0
        self.low_confidence_translations = 0

    def record_issue(self, issue: TranslationIssue):
        """Record a translation issue"""
        self.issues.append(issue)

        if issue.severity == IssueSeverity.ERROR:
            self.failed_translations += 1
        elif issue.confidence < 0.5:
            self.low_confidence_translations += 1

    def record_success(self, source_text: str, translated_text: str,
                       confidence: float, location: str):
        """Record a successful translation"""
        self.total_translations += 1

        if confidence >= 0.85:
            self.successful_translations += 1
        elif confidence < 0.5:
            # Low confidence - record as issue
            self.record_issue(TranslationIssue(
                issue_type=IssueType.LOW_CONFIDENCE,
                severity=IssueSeverity.WARNING,
                source_text=source_text,
                translated_text=translated_text,
                confidence=confidence,
                location=location
            ))
            self.low_confidence_translations += 1
        else:
            self.successful_translations += 1

    def record_failure(self, source_text: str, location: str,
                       error_message: str, attempted_strategies: List[str]):
        """Record a translation failure"""
        self.total_translations += 1
        self.failed_translations += 1

        self.record_issue(TranslationIssue(
            issue_type=IssueType.NO_TRANSLATION,
            severity=IssueSeverity.ERROR,
            source_text=source_text,
            translated_text=None,
            confidence=0.0,
            location=location,
            attempted_strategies=attempted_strategies,
            error_message=error_message
        ))

    def get_summary(self) -> Dict[str, Any]:
        """Get translation summary statistics"""
        return {
            "total_translations": self.total_translations,
            "successful": self.successful_translations,
            "failed": self.failed_translations,
            "low_confidence": self.low_confidence_translations,
            "success_rate": (self.successful_translations / self.total_translations * 100)
                           if self.total_translations > 0 else 0,
            "total_issues": len(self.issues),
            "issues_by_type": self._count_by_type(),
            "issues_by_severity": self._count_by_severity()
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count issues by type"""
        counts = {}
        for issue in self.issues:
            issue_type = issue.issue_type.value
            counts[issue_type] = counts.get(issue_type, 0) + 1
        return counts

    def _count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity"""
        counts = {}
        for issue in self.issues:
            severity = issue.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def save_report(self, output_path: Path):
        """Save detailed issue report to JSON file"""
        report = {
            "summary": self.get_summary(),
            "issues": [issue.to_dict() for issue in self.issues]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("TRANSLATION QUALITY REPORT")
        print("=" * 80)
        print(f"Total Translations: {summary['total_translations']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Low Confidence: {summary['low_confidence']}")
        print(f"\nTotal Issues: {summary['total_issues']}")

        if summary['issues_by_severity']:
            print("\nIssues by Severity:")
            for severity, count in summary['issues_by_severity'].items():
                print(f"  {severity.upper()}: {count}")

        if summary['issues_by_type']:
            print("\nIssues by Type:")
            for issue_type, count in summary['issues_by_type'].items():
                print(f"  {issue_type}: {count}")

        print("=" * 80)
