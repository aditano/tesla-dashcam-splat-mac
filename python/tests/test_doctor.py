from dashsplat.doctor import run_doctor


def test_doctor_report_has_host_fields(capsys):
    report = run_doctor(as_json=True)
    assert "host" in report
    assert report["host"]["ram_bytes"] > 0
    assert "preferences" in report
    assert "decode" in report["preferences"]
    out = capsys.readouterr().out
    assert "ram" in out.lower() or "host" in out or "doctor" in out
