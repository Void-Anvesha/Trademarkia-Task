from app.core.pipeline import build_artifacts


if __name__ == "__main__":
    report = build_artifacts(force_rebuild=True)
    print("Build complete")
    print(f"Documents: {report['num_documents']}")
    print(f"Selected clusters: {report['selected_clusters']}")
