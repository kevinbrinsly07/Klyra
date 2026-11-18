
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

content = [
    "LankaMind AI Knowledge Base",
    "",
    "Sri Lanka Driving Licence Information",
    "",
    "To obtain a driving licence in Sri Lanka, an applicant must",
    "first obtain a learner's permit. Applicants must meet the",
    "minimum age requirements and provide valid identification.",
    "",
    "The applicant must pass a written examination before",
    "becoming eligible for the practical driving test.",
    "",
    "After successfully passing both examinations, the applicant",
    "can obtain a driving licence from the Department of Motor Traffic.",
    "",
    "Sri Lanka Education Information",
    "",
    "Students in Sri Lanka can pursue Ordinary Level (O/L) and",
    "Advanced Level (A/L) examinations. The Advanced Level",
    "examination is generally used for university admission.",
    "",
    "Sri Lanka has both public and private universities offering",
    "undergraduate and postgraduate programmes."
]

pdf = canvas.Canvas("sample_knowledge.pdf", pagesize=A4)

width, height = A4
y = height - 60

for line in content:
    if y < 60:
        pdf.showPage()
        y = height - 60

    pdf.drawString(50, y, line)
    y -= 20

pdf.save()

print("Sample PDF created successfully!")
