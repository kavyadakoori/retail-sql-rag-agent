from database import load_data as loader


def test_loader_uses_configured_table_name(tmp_path, monkeypatch):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("id\n1\n", encoding="utf-8")
    executed = {}

    class FakeCursor:
        def executemany(self, query, rows):
            executed["query"] = query
            executed["rows"] = rows

        def close(self):
            pass

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            pass

        def close(self):
            pass

    connection = FakeConnection()
    monkeypatch.setattr(loader, "get_db_config", lambda: {})
    monkeypatch.setattr(loader, "create_tables", lambda connection: None)
    monkeypatch.setattr(
        loader,
        "load_table_rows",
        lambda table_name, columns, path: [(1,)],
    )
    monkeypatch.setattr(
        loader,
        "TABLE_CONFIG",
        {
            "records": {
                "file": csv_path,
                "columns": ["id"],
            }
        },
    )
    monkeypatch.setattr(
        loader.mysql.connector,
        "connect",
        lambda **kwargs: connection,
    )

    loader.load_data()

    assert "INSERT INTO records" in executed["query"]
    assert executed["rows"] == [(1,)]
