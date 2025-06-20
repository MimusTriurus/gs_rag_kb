import re
import sys
import json
import yaml
from rake_nltk import Rake
from transformers import pipeline


def extract_front_matter(md_content):
    """
    Extracts YAML front matter from the Markdown content.
    Returns a dictionary with metadata if found, otherwise an empty dictionary.
    """
    front_matter_regex = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
    match = front_matter_regex.match(md_content)
    if match:
        try:
            data = yaml.safe_load(match.group(1))
            return data
        except yaml.YAMLError as e:
            print("YAML parsing error:", e)
    return {}


def extract_headings(md_content, levels=(1, 2, 3)):
    """
    Extracts Markdown headings from content for the specified heading levels.

    :param md_content: The full text content of the Markdown file.
    :param levels: A tuple of heading levels to extract (default is 1, 2, and 3).
    :return: A list of tuples, each containing (heading level, heading text).
    """
    pattern = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
    headings = set()
    for match in pattern.finditer(md_content):
        level = len(match.group(1))
        if level in levels:
            headings.add((level, match.group(2).strip()))
    return list(headings)


def extract_keywords_rake(md_content):
    """
    Uses RAKE (Rapid Automatic Keyword Extraction) to extract key phrases from the text.

    :param md_content: The input text content.
    :return: A list of keyword phrases.
    """
    r = Rake()  # Initialize RAKE with default stopwords
    r.extract_keywords_from_text(md_content)
    keywords_with_scores = r.get_ranked_phrases_with_scores()
    # Return only the phrases (ignoring their scores)
    return [phrase for score, phrase in keywords_with_scores]


def summarize_text(md_content, max_length=150, min_length=40):
    """
    Summarizes the given Markdown content using a transformer-based summarization pipeline.

    :param md_content: The full text content.
    :param max_length: Maximum length of the summary.
    :param min_length: Minimum length of the summary.
    :return: A summarized version of the text.
    """
    summarizer = pipeline("summarization")
    # Note: For very long text, consider splitting the content into chunks.
    summary = summarizer(md_content, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def extract_document_metadata(md_file_path):
    """
    Extracts metadata from a Markdown file by combining:
      - YAML front matter (if available)
      - Markdown headings
      - Keywords extracted via RAKE
      - A generated summary of the document

    :param md_file_path: Path to the Markdown file.
    :return: A dictionary containing the combined metadata.
    """
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {md_file_path}: {e}")
        sys.exit(1)

    metadata = {}

    # 1. Extract front matter (if exists)
    #front_matter = extract_front_matter(content)
    #if front_matter:
    #    metadata.update(front_matter)

    # 2. Extract headings from the document
    headings = extract_headings(content)
    metadata["headings"] = headings

    # 3. Extract keywords using RAKE
    #metadata["keywords"] = extract_keywords_rake(content)

    # 4. Generate a summary of the document
    try:
        summary = summarize_text(content)
        metadata["summary"] = summary
    except Exception as e:
        print("Summarization error:", e)
        metadata["summary"] = "Summarization failed."

    return metadata


def main():
    md_file_path = 'documents_no_meta/connect_2_local_dedic.md'
    metadata = extract_document_metadata(md_file_path)

    # Print the extracted metadata in JSON format for readability
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
