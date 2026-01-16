#First execute this
import os
import pandas as pd

try:
    import PyPDF2
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False

data_path = 'data/data'
output_resumes = 'data/resumes_clean.csv'
output_labels = 'data/labels_clean.csv'

# Get all category folders
categories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

print(f"Found {len(categories)} job categories")

resumes_data = []
labels_data = []
resume_count = 0

for category in sorted(categories):
    cat_path = os.path.join(data_path, category)
    resume_files = [f for f in os.listdir(cat_path) if f.endswith(('.txt', '.pdf'))]
    
    print(f"✓ Processing {category}: {len(resume_files)} resumes")
    
    for file_idx, resume_file in enumerate(resume_files):
        file_path = os.path.join(cat_path, resume_file)
        
        try:
            resume_text = ""
            
            # Extract text from TXT
            if resume_file.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
            
            # Extract text from PDF
            elif resume_file.endswith('.pdf') and PDF_SUPPORT:
                try:
                    pdf_reader = PyPDF2.PdfReader(file_path)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            resume_text += page_text + " "
                except:
                    print(f"  ✗ Failed to extract PDF: {resume_file}")
                    continue
            
            # Clean text
            resume_text = resume_text.strip()
            
            # Skip if empty
            if len(resume_text) < 50:
                print(f"  ✗ Skipped {resume_file} (too short)")
                continue
            
            resume_id = f"{category}_{file_idx}"
            
            # Add to resumes data
            resumes_data.append({
                'resume_id': resume_id,
                'category': category,
                'resume_text': resume_text[:5000]
            })
            
            # Create score based on text length
            score = 50
            if len(resume_text) > 500:
                score += 15
            if len(resume_text) > 1000:
                score += 15
            if len(resume_text) > 2000:
                score += 10
            
            score = min(score, 100)
            
            labels_data.append({
                'resume_id': resume_id,
                'category': category,
                'score': score
            })
            
            resume_count += 1
            
        except Exception as e:
            print(f"  ✗ Error reading {resume_file}: {e}")
            continue

# Save to CSV
resumes_df = pd.DataFrame(resumes_data)
resumes_df.to_csv(output_resumes, index=False)

labels_df = pd.DataFrame(labels_data)
labels_df.to_csv(output_labels, index=False)

print(f"\n✅ Successfully processed {resume_count} resumes!")
print(f"✅ Created {output_resumes}")
print(f"✅ Created {output_labels}")
print(f"Score range: {labels_df['score'].min()} - {labels_df['score'].max()}")
