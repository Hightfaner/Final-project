import re

from src.sanitiser import meaningful_body, sanitise_text


def test_active_html_is_removed_and_never_executed():
    result = sanitise_text("<script>alert('x')</script><p>Safe!</p>")
    assert "alert" not in result.text
    assert "Safe!" in result.text
    assert result.removed_active_content_count == 1


def test_urls_become_static_host_path_markers():
    result = sanitise_text("Visit https://Example.com/login?token=secret now")
    assert "https://" not in result.text
    assert "token=secret" not in result.text
    assert "[URL: example.com/login]" in result.text
    assert result.url_replacement_count == 1


def test_anchor_and_remote_image_are_safely_rendered():
    result = sanitise_text(
        '<a href="https://example.org/a">Review</a><img src="https://tracker.test/x.png">'
    )
    assert "Review [URL: example.org/a]" in result.text
    assert "[REMOTE IMAGE REMOVED]" in result.text
    assert "tracker.test" not in result.text


def test_email_and_phone_pii_are_replaced_conservatively():
    result = sanitise_text("Contact Alice.Person@example.com or +44 20 7946 0958")
    assert "Alice.Person" not in result.text
    assert "[USER]@example.com" in result.text
    assert "[PHONE]" in result.text
    assert result.pii_replacement_count == 2


def test_sanitisation_is_idempotent_and_preserves_evidence():
    first = sanitise_text("URGENT!! &amp;amp; visit www.example.com/a")
    second = sanitise_text(first.text)
    assert first.text == second.text
    assert "URGENT!!" in first.text


def test_adjacent_phone_like_sequences_reach_a_fixed_point():
    first = sanitise_text("Cel: +44 20 7946 0308\nBiper: 123 456 7890 cve 801\t9581")
    second = sanitise_text(first.text)
    assert first.text == second.text


def test_plain_angle_bracket_decoration_is_not_treated_as_html():
    first = sanitise_text("<<<< Hi user >>>> Please reply <<<<")
    second = sanitise_text(first.text)
    assert first.text == second.text
    assert "<<<<" in first.text


def test_meaningful_body_uses_only_the_label_independent_blank_rule():
    assert not meaningful_body(" \n\t ")
    assert meaningful_body("[REMOTE IMAGE REMOVED]\n[URL: example.com/]")
    assert meaningful_body("عزيزي العميل")
