#!/usr/bin/env python3
"""
Create Sample PDF for Translation Demonstration
Generates a comprehensive PDF with various content types
"""

import fitz
import sys
from pathlib import Path

def create_sample_pdf(output_path):
    """Create a comprehensive sample PDF"""

    doc = fitz.open()

    # ========================================
    # PAGE 1: SCIENTIFIC RESEARCH PAPER
    # ========================================
    page1 = doc.new_page(width=595, height=842)

    # Title
    page1.insert_text((50, 50), "Quantum Computing Research Paper", fontsize=20)
    page1.insert_text((50, 80), "Advanced Studies in Quantum Mechanics", fontsize=14)

    # Author
    page1.insert_text((50, 120), "Dr. John Smith, MIT Laboratory", fontsize=12)
    page1.insert_text((50, 140), "Published: March 2025", fontsize=10)

    # Abstract
    page1.insert_text((50, 180), "Abstract", fontsize=14)
    abstract = """This research investigates the quantum properties of photonic
crystals and their applications in quantum computing. Our study
demonstrates significant improvements in quantum coherence time
and error correction capabilities."""

    y = 210
    for line in abstract.split('\n'):
        page1.insert_text((50, y), line.strip(), fontsize=11)
        y += 20

    # Introduction
    page1.insert_text((50, 300), "1. Introduction", fontsize=14)
    intro = """Quantum computing represents a paradigm shift in computational
theory and practice. The fundamental principle relies on quantum
superposition and entanglement to perform calculations exponentially
faster than classical computers for certain problem classes."""

    y = 330
    for line in intro.split('\n'):
        page1.insert_text((50, y), line.strip(), fontsize=11)
        y += 20

    # Mathematical formulas
    page1.insert_text((50, 440), "2. Theoretical Framework", fontsize=14)
    page1.insert_text((50, 470), "The quantum state is described by:", fontsize=11)
    page1.insert_text((100, 500), "|ψ⟩ = α|0⟩ + β|1⟩", fontsize=12)
    page1.insert_text((50, 530), "where |α|² + |β|² = 1", fontsize=11)

    # Energy equation
    page1.insert_text((50, 570), "The Hamiltonian energy:", fontsize=11)
    page1.insert_text((100, 600), "E = ℏω(n + 1/2)", fontsize=12)

    # Page number
    page1.insert_text((270, 800), "Page 1", fontsize=10, color=(0.5, 0.5, 0.5))

    # ========================================
    # PAGE 2: EXPERIMENTAL RESULTS
    # ========================================
    page2 = doc.new_page(width=595, height=842)

    page2.insert_text((50, 50), "3. Experimental Results", fontsize=16)

    # Results section
    results = """Our experiments demonstrate a coherence time of 150 microseconds,
representing a 40% improvement over previous results. The quantum
gate fidelity achieved was 99.7%, exceeding the fault-tolerant
threshold required for practical quantum computing."""

    y = 100
    for line in results.split('\n'):
        page2.insert_text((50, y), line.strip(), fontsize=11)
        y += 20

    # Table header
    page2.insert_text((50, 200), "Table 1: Experimental Measurements", fontsize=13)

    # Draw table
    table_rect = fitz.Rect(50, 220, 545, 380)
    page2.draw_rect(table_rect, color=(0, 0, 0), width=1)

    # Table headers
    page2.draw_line((50, 250), (545, 250), color=(0, 0, 0), width=1)
    page2.draw_line((200, 220), (200, 380), color=(0, 0, 0), width=1)
    page2.draw_line((380, 220), (380, 380), color=(0, 0, 0), width=1)

    page2.insert_text((80, 240), "Parameter", fontsize=11)
    page2.insert_text((240, 240), "Value", fontsize=11)
    page2.insert_text((410, 240), "Unit", fontsize=11)

    # Table data
    table_data = [
        ("Coherence Time", "150", "μs", 270),
        ("Gate Fidelity", "99.7", "%", 300),
        ("Qubit Count", "127", "qubits", 330),
        ("Error Rate", "0.003", "per gate", 360),
    ]

    for param, value, unit, y_pos in table_data:
        page2.insert_text((60, y_pos), param, fontsize=10)
        page2.insert_text((240, y_pos), value, fontsize=10)
        page2.insert_text((410, y_pos), unit, fontsize=10)

    # Discussion
    page2.insert_text((50, 420), "4. Discussion", fontsize=14)
    discussion = """These results indicate significant progress in quantum computing
hardware. The improved coherence time enables more complex quantum
algorithms to be executed before decoherence occurs."""

    y = 450
    for line in discussion.split('\n'):
        page2.insert_text((50, y), line.strip(), fontsize=11)
        y += 20

    # Watermark
    page2.insert_text((200, 600), "CONFIDENTIAL", fontsize=60,
                     color=(0.9, 0.9, 0.9))

    page2.insert_text((270, 800), "Page 2", fontsize=10, color=(0.5, 0.5, 0.5))

    # ========================================
    # PAGE 3: CONCLUSION
    # ========================================
    page3 = doc.new_page(width=595, height=842)

    page3.insert_text((50, 50), "5. Conclusion", fontsize=16)

    conclusion = """In conclusion, our research demonstrates substantial improvements
in quantum computing hardware performance. The 40% increase in
coherence time and 99.7% gate fidelity represent significant
milestones toward practical quantum computing applications.

Future work will focus on scaling up the qubit count while
maintaining these performance metrics, and developing error
correction protocols optimized for our hardware architecture."""

    y = 100
    for line in conclusion.split('\n'):
        if line.strip():
            page3.insert_text((50, y), line.strip(), fontsize=11)
        y += 20

    # References
    page3.insert_text((50, 250), "References", fontsize=14)

    references = [
        "[1] Smith, J. et al. (2024). Quantum coherence in photonic systems.",
        "    Nature Physics, 20(3), 234-245.",
        "",
        "[2] Johnson, M. (2024). Advances in quantum error correction.",
        "    Physical Review Letters, 132(8), 080501.",
        "",
        "[3] Chen, L. et al. (2023). Scalable quantum computing architectures.",
        "    Science, 381(6654), 1234-1238.",
    ]

    y = 280
    for ref in references:
        page3.insert_text((50, y), ref, fontsize=10)
        y += 18

    # Acknowledgments
    page3.insert_text((50, 450), "Acknowledgments", fontsize=14)
    ack = """This work was supported by the National Science Foundation
under Grant No. PHY-2024001. We thank the MIT Quantum Computing
Laboratory for providing access to experimental facilities."""

    y = 480
    for line in ack.split('\n'):
        page3.insert_text((50, y), line.strip(), fontsize=10)
        y += 18

    # Contact
    page3.insert_text((50, 600), "Contact Information", fontsize=14)
    page3.insert_text((50, 630), "Email: john.smith@mit.edu", fontsize=10)
    page3.insert_text((50, 650), "Lab: MIT Quantum Computing Lab", fontsize=10)
    page3.insert_text((50, 670), "Website: https://quantumlab.mit.edu", fontsize=10)

    page3.insert_text((270, 800), "Page 3", fontsize=10, color=(0.5, 0.5, 0.5))

    # Save PDF
    doc.save(output_path)
    doc.close()

    return output_path


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "demonstration/input/sample_paper.pdf"

    result = create_sample_pdf(output_path)
    print(f"✓ Created sample PDF: {result}")

    # Print statistics
    doc = fitz.open(result)
    print(f"  Pages: {len(doc)}")
    print(f"  Size: {Path(result).stat().st_size / 1024:.1f} KB")

    # Count words
    total_words = 0
    for page in doc:
        text = page.get_text()
        total_words += len(text.split())

    print(f"  Words: ~{total_words}")
    doc.close()
