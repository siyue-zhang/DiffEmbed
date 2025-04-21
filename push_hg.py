from huggingface_hub import create_repo

repo_url = create_repo("Dream_emb", private=False)

from huggingface_hub import upload_folder

upload_folder(
    repo_id="siyue/Dream_emb",
    folder_path="/home/siyue001/Projects/llm2vec_reason_dream/dream",
    path_in_repo=".",  # or "models/", "weights/", etc.
)
