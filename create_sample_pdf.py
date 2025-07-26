#!/usr/bin/env python3
"""
Create a sample PDF for testing the RAG chatbot
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_sample_pdf():
    """Create a sample PDF with science content for kids"""
    
    # Read the content
    with open("sample_content.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate("sample_knowledge_base.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=12,
        textColor='blue'
    )
    
    chapter_style = ParagraphStyle(
        'ChapterTitle',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=8,
        textColor='darkblue'
    )
    
    # Build story
    story = []
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
            
        if line == "Science Learning Guide for Kids":
            story.append(Paragraph(line, title_style))
            story.append(Spacer(1, 12))
        elif line.startswith("Chapter"):
            story.append(Spacer(1, 12))
            story.append(Paragraph(line, chapter_style))
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 4))
    
    # Build PDF
    doc.build(story)
    print("✅ Sample PDF created: sample_knowledge_base.pdf")

if __name__ == "__main__":
    create_sample_pdf()