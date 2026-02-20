import os

from legal_rag_system import LegalRAGSystem


def main() -> None:
    docs_dir = "documents"
    pdfs = []
    txts = []
    for name in os.listdir(docs_dir):
        path = os.path.join(docs_dir, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if low.endswith(".pdf"):
            pdfs.append(path)
        elif low.endswith(".txt"):
            txts.append(path)

    pdfs.sort()
    txts.sort()
    print("PDF_COUNT", len(pdfs))
    print("TXT_COUNT", len(txts))

    rag = LegalRAGSystem()
    ok, fail = rag.load_documents(pdf_paths=pdfs, txt_paths=txts)
    print("DONE_OK", len(ok), "DONE_FAIL", len(fail))
    if fail:
        print("FAIL_LIST")
        for item in fail:
            print(item)


if __name__ == "__main__":
    main()
