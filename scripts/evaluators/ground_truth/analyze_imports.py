#!/usr/bin/env python3
"""
Import Analysis Tool for Orchestration Files

Analyzes Python files to discover all imported modules and packages,
categorizes them, and generates a detailed report to help decide which
dependencies to install for evaluation environments.

Usage:
    python analyze_imports.py --input-dir mutation_benchmark/originals \
        --output report_imports.json \
        --orchestrators airflow prefect dagster
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Any, Optional


# Standard library modules (Python 3.8+)
STDLIB_MODULES = {
    # Core
    '__future__', 'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
    'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect',
    'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code',
    'codecs', 'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
    'contextlib', 'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes',
    'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils',
    'doctest', 'email', 'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp',
    'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
    'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac',
    'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
    'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
    'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder',
    'msilib', 'msvcrt', 'multiprocessing', 'netrc', 'nis', 'nntplib', 'numbers', 'operator',
    'optparse', 'os', 'ossaudiodev', 'parser', 'pathlib', 'pdb', 'pickle', 'pickletools',
    'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
    'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri',
    'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd',
    'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl', 'stat', 'statistics',
    'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig',
    'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
    'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace', 'traceback',
    'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing', 'typing_extensions',
    'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref',
    'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp',
    'zipfile', 'zipimport', 'zlib', 'zoneinfo',
}

# Known orchestrator frameworks
ORCHESTRATOR_PACKAGES = {
    'airflow': 'Apache Airflow',
    'prefect': 'Prefect',
    'dagster': 'Dagster',
}

# Common data science / processing packages
COMMON_PACKAGES = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'scipy': 'Scientific computing',
    'sklearn': 'Machine learning (scikit-learn)',
    'tensorflow': 'Machine learning',
    'torch': 'Machine learning (PyTorch)',
    'keras': 'Machine learning',
    'matplotlib': 'Plotting',
    'seaborn': 'Statistical visualization',
    'plotly': 'Interactive plotting',
    'requests': 'HTTP library',
    'httpx': 'HTTP client',
    'aiohttp': 'Async HTTP',
    'beautifulsoup4': 'HTML/XML parsing (bs4)',
    'lxml': 'XML/HTML processing',
    'sqlalchemy': 'SQL toolkit',
    'psycopg2': 'PostgreSQL adapter',
    'pymongo': 'MongoDB driver',
    'redis': 'Redis client',
    'celery': 'Task queue',
    'pydantic': 'Data validation',
    'fastapi': 'Web framework',
    'flask': 'Web framework',
    'django': 'Web framework',
    'boto3': 'AWS SDK',
    'google': 'Google Cloud',
    'azure': 'Azure SDK',
    'pyspark': 'Spark',
    'dask': 'Parallel computing',
    'polars': 'DataFrame library',
    'pyarrow': 'Apache Arrow',
    'openpyxl': 'Excel files',
    'xlrd': 'Excel files (legacy)',
    'pytest': 'Testing',
    'pytest-cov': 'Test coverage',
    'black': 'Code formatter',
    'flake8': 'Linting',
    'mypy': 'Type checking',
    'jinja2': 'Templating',
    'pyyaml': 'YAML parser',
    'toml': 'TOML parser',
    'click': 'CLI framework',
    'typer': 'CLI framework',
    'rich': 'Terminal formatting',
    'tqdm': 'Progress bars',
    'python-dotenv': 'Environment variables',
    'cryptography': 'Cryptographic recipes',
}


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    module: str
    is_from: bool
    imported_names: List[str]
    file_path: str
    line_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleStats:
    """Statistics for a single module."""
    module: str
    category: str  # stdlib, orchestrator, common, unknown
    occurrences: int = 0
    files: Set[str] = field(default_factory=set)
    import_statements: List[ImportInfo] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'module': self.module,
            'category': self.category,
            'occurrences': self.occurrences,
            'file_count': len(self.files),
            'files': sorted(list(self.files)),
            'sample_imports': [imp.to_dict() for imp in self.import_statements[:5]],
        }


@dataclass
class OrchestratorReport:
    """Report for a single orchestrator."""
    orchestrator: str
    total_files: int = 0
    parseable_files: int = 0
    failed_files: int = 0
    modules: Dict[str, ModuleStats] = field(default_factory=dict)
    
    def add_import(self, import_info: ImportInfo) -> None:
        """Add an import to the report."""
        module = import_info.module
        
        if module not in self.modules:
            # Categorize the module
            category = self._categorize_module(module)
            self.modules[module] = ModuleStats(module=module, category=category)
        
        stats = self.modules[module]
        stats.occurrences += 1
        stats.files.add(import_info.file_path)
        stats.import_statements.append(import_info)
    
    def _categorize_module(self, module: str) -> str:
        """Categorize a module as stdlib, orchestrator, common, or unknown."""
        root = module.split('.')[0]
        
        if root in STDLIB_MODULES:
            return 'stdlib'
        if root in ORCHESTRATOR_PACKAGES:
            return 'orchestrator'
        if root in COMMON_PACKAGES or module in COMMON_PACKAGES:
            return 'common'
        return 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'orchestrator': self.orchestrator,
            'summary': {
                'total_files': self.total_files,
                'parseable_files': self.parseable_files,
                'failed_files': self.failed_files,
                'unique_modules': len(self.modules),
            },
            'modules_by_category': self._group_by_category(),
            'top_missing_dependencies': self._get_top_missing(20),
        }
    
    def _group_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group modules by category."""
        grouped = defaultdict(list)
        for stats in self.modules.values():
            grouped[stats.category].append(stats.to_dict())
        
        # Sort each category by occurrence count
        for category in grouped:
            grouped[category].sort(key=lambda x: x['occurrences'], reverse=True)
        
        return dict(grouped)
    
    def _get_top_missing(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get top N non-stdlib modules by occurrence."""
        non_stdlib = [
            stats for stats in self.modules.values()
            if stats.category in ('common', 'unknown')
        ]
        non_stdlib.sort(key=lambda x: x.occurrences, reverse=True)
        return [s.to_dict() for s in non_stdlib[:n]]


class ImportAnalyzer:
    """Analyzes Python files to extract import statements."""
    
    def __init__(self):
        self.reports: Dict[str, OrchestratorReport] = {}
    
    def analyze_file(self, file_path: Path, orchestrator: str) -> Optional[List[ImportInfo]]:
        """
        Analyze a single Python file and extract all imports.
        
        Returns:
            List of ImportInfo objects, or None if file cannot be parsed
        """
        try:
            code = file_path.read_text(encoding='utf-8')
            tree = ast.parse(code)
        except SyntaxError:
            return None
        except Exception as e:
            print(f"[WARN] Failed to read {file_path}: {e}", file=sys.stderr)
            return None
        
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        is_from=False,
                        imported_names=[alias.asname or alias.name],
                        file_path=str(file_path.name),
                        line_number=node.lineno,
                    ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names = [alias.name for alias in node.names]
                    imports.append(ImportInfo(
                        module=node.module,
                        is_from=True,
                        imported_names=imported_names,
                        file_path=str(file_path.name),
                        line_number=node.lineno,
                    ))
        
        return imports
    
    def analyze_directory(self, directory: Path, orchestrator: str) -> OrchestratorReport:
        """Analyze all Python files in a directory for a given orchestrator."""
        report = OrchestratorReport(orchestrator=orchestrator)
        
        files = sorted(directory.glob('*.py'))
        report.total_files = len(files)
        
        for file_path in files:
            imports = self.analyze_file(file_path, orchestrator)
            
            if imports is None:
                report.failed_files += 1
                continue
            
            report.parseable_files += 1
            
            for import_info in imports:
                report.add_import(import_info)
        
        return report
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report across all orchestrators."""
        # Aggregate statistics
        total_files = sum(r.total_files for r in self.reports.values())
        total_parseable = sum(r.parseable_files for r in self.reports.values())
        
        # Find most common non-stdlib modules across all orchestrators
        all_modules: Counter = Counter()
        for report in self.reports.values():
            for stats in report.modules.values():
                if stats.category in ('common', 'unknown'):
                    all_modules[stats.module] += stats.occurrences
        
        return {
            'summary': {
                'total_files': total_files,
                'parseable_files': total_parseable,
                'failed_files': total_files - total_parseable,
                'orchestrators_analyzed': list(self.reports.keys()),
            },
            'top_missing_dependencies_global': [
                {'module': mod, 'total_occurrences': count}
                for mod, count in all_modules.most_common(30)
            ],
            'by_orchestrator': {
                orch: report.to_dict()
                for orch, report in self.reports.items()
            },
            'installation_recommendations': self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate installation recommendations for each orchestrator."""
        recommendations = {}
        
        for orch, report in self.reports.items():
            # Get top non-stdlib modules
            top_missing = []
            for stats in report.modules.values():
                if stats.category in ('common', 'unknown'):
                    top_missing.append((stats.module, stats.occurrences, len(stats.files)))
            
            top_missing.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
            # Generate pip install commands
            packages = []
            for module, occ, files in top_missing[:15]:
                # Try to map import name to package name
                pkg_name = self._module_to_package(module)
                if pkg_name:
                    packages.append(f"{pkg_name}  # used in {files} files ({occ} imports)")
            
            recommendations[orch] = packages
        
        return recommendations
    
    def _module_to_package(self, module: str) -> Optional[str]:
        """Map import module name to pip package name."""
        # Direct mappings for common cases
        mappings = {
            'bs4': 'beautifulsoup4',
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'yaml': 'pyyaml',
            'dotenv': 'python-dotenv',
        }
        
        root = module.split('.')[0]
        
        if root in mappings:
            return mappings[root]
        
        # For most packages, the import name matches the package name
        if root in COMMON_PACKAGES:
            return root
        
        # Return the root module name as best guess
        return root


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Python files to discover imported modules and dependencies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        type=Path,
        help='Directory containing orchestrator subdirectories (e.g., mutation_benchmark/originals)',
    )
    
    parser.add_argument(
        '--orchestrators',
        nargs='+',
        default=['airflow', 'prefect', 'dagster'],
        help='Orchestrators to analyze',
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('import_analysis_report.json'),
        help='Output JSON report path',
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress',
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    analyzer = ImportAnalyzer()
    
    print("Analyzing imports...")
    print("=" * 60)
    
    for orchestrator in args.orchestrators:
        orch_dir = input_dir / orchestrator
        
        if not orch_dir.exists():
            print(f"[WARN] Directory not found: {orch_dir}")
            continue
        
        if args.verbose:
            print(f"\nAnalyzing {orchestrator}...")
        
        report = analyzer.analyze_directory(orch_dir, orchestrator)
        analyzer.reports[orchestrator] = report
        
        print(f"[{orchestrator}] {report.parseable_files}/{report.total_files} files parsed")
        print(f"[{orchestrator}] {len(report.modules)} unique modules found")
    
    # Generate and save report
    print("\nGenerating report...")
    full_report = analyzer.generate_report()
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = full_report['summary']
    print(f"Total files analyzed: {summary['total_files']}")
    print(f"Successfully parsed: {summary['parseable_files']}")
    print(f"Failed to parse: {summary['failed_files']}")
    
    print("\nTop missing dependencies (across all orchestrators):")
    for item in full_report['top_missing_dependencies_global'][:10]:
        print(f"  • {item['module']}: {item['total_occurrences']} imports")
    
    print("\nInstallation recommendations:")
    for orch, packages in full_report['installation_recommendations'].items():
        print(f"\n  [{orch.upper()}] Top packages to install:")
        for pkg in packages[:10]:
            print(f"    {pkg}")
    
    print("\n" + "=" * 60)
    print(f"Full report available at: {args.output}")


if __name__ == '__main__':
    main()