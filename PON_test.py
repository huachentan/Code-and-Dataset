import os
from openai import OpenAI
from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def compare_docs_with_qwen(doc1_path, doc2_path, output_folder):
    try:

        doc1_content = read_docx(doc1_path)
        doc2_content = read_docx(doc2_path)


        client = OpenAI(
            api_key="Your_API_Key",
            base_url="Your_Base_URL",
        )

        # 构建prompt
        prompt = f"""
        Please perform a multi-dimensional text comparison analysis of these two texts to determine the risk of plagiarism according to the hierarchy rules:

        I. Structural elements detection
        1. Narrative structure analysis (weight 0.15)
        (1) Comparison items: main logic chain/scene sequence/timeline organization/sequence of events
        (2) Judgment line: similarity ≥70% → risk marker*

        2. Character system analysis (weight 0.35)
        (1) Comparison items: character profile/character matrix/relationship network/growth trajectory
        (2) Judgment line: similarity ≥50%→risk marking*

        3. Plot content analysis (weight 0.5)
        (1) Comparison items: plot topology map / development rhythm / chain of key events / descriptive statements
        (2) Judgment line: similarity ≥30%→risk marker*

        II. Comprehensive Judgment Rules
        1. If any structured element triggers the risk marker*, the risk of plagiarism will be determined directly.
        2. Calculation when the mark is not triggered: comprehensive similarity = (narrative structure value * 0.15) + (character system value * 0.35) + (plot content value * 0.5) → comprehensive value ≥ 30% plagiarism risk determination

        III. Literal detection rules (independently effective)
        1. Text coverage: literal repetitions ≥ 30%
        2. Continuous repetition: the existence of ≥10 words of identical fragments
        Risk is determined if any condition is met

        IV. Exclusion mechanism
        The following common elements are not counted as similarity:
        ① Abstract themes (e.g., macro concepts such as justice, war, etc.)
        ② Types of subject matter (e.g., categorized labels such as sci-fi, suspense, etc.)
        ③ Genres of writing (novels/poetry and other forms)
        ④ Public knowledge (scientific principles, historical facts, etc.)

        Document1:
        {doc1_content}

        Document2:
        {doc2_content}

        Output Format:
        [Testing Conclusion] 
        - Sub-similarity:
            Narrative structure: __% 
            Character system: __%    
            Plot content: __% 
        - Overall similarity: __% (weighted calculated value) 
        - Verdict: [Risk of plagiarism / No risk of plagiarism] 
        [Examples of Evidence] 
        1. Example of structural repetition:
            “Paragraph X of the text to be examined: ... ←→ Reference text paragraph Y: ...”  
        2. Role cloning evidence:     
            “Character A's [occupation/mantra/behavioral pattern] is highly similar to that of reference character B.” 
        3. Evidence of episode similarity:
            “Episode A to be examined is highly similar to Reference Episode B” 
        4. Literal repetitive passages: 
            “Repeat of 10 words: '_________' (Location: P3 to be examined vs. Reference P8)”
        """

        completion = client.chat.completions.create(
            model="Your_Model_ID",  
            messages=[
                {'role': 'system', 'content': 'You are a legal expert specializing in copyright plagiarism and similarity detection.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.5  
        )

        response = completion.choices[0].message.content
        print(f"Similarity Result for {doc2_path}: {response}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        doc2_filename = os.path.basename(doc2_path)
        output_file_path = os.path.join(output_folder, f"result_{os.path.splitext(doc2_filename)[0]}.docx")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(response)

        print(f"The results have been saved to: {output_file_path}")

    except Exception as e:
        print(f"Error message for {doc2_path}: {e}")
        print("Please refer to the documentation: Your_Documentation Link")

doc1_path = r'Example of PlagiarismCheck\Originaltext1.docx'
comparative_folder = r'Example of PlagiarismCheck\Comparativetext1'
output_folder = r'Example of PlagiarismCheck\LLMresult1'

for filename in os.listdir(comparative_folder):
    if filename.endswith('.docx'):
        doc2_path = os.path.join(comparative_folder, filename)
        compare_docs_with_qwen(doc1_path, doc2_path, output_folder)