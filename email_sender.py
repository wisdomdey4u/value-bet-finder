"""
email_sender.py
----------------
Composes and sends the daily value-bet email via Gmail SMTP, with retry
and exponential backoff on failure.
"""

import time
import smtplib
import logging
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config

logger = logging.getLogger("value_bet_finder.email_sender")


def _format_pick_html(rank: int, pick: dict) -> str:
    edge_pct = f"{pick['edge']*100:.1f}%" if pick.get("edge") is not None else "n/a"
    ev_pct = f"{pick['expected_value']*100:.1f}%"
    kickoff = pick["commence_time"]
    try:
        kickoff_dt = dt.datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        kickoff = kickoff_dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    return f"""
    <div style="border:1px solid #e0e0e0;border-radius:8px;padding:16px;margin-bottom:14px;
                font-family:Arial,Helvetica,sans-serif;">
      <div style="font-size:16px;font-weight:bold;color:#1a1a1a;margin-bottom:6px;">
        Pick #{rank}: {pick['home_team']} vs {pick['away_team']}
      </div>
      <div style="color:#555;font-size:13px;margin-bottom:10px;">
        League: {pick['league']} &nbsp;|&nbsp; Kickoff: {kickoff}
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <tr><td style="padding:4px 0;color:#777;">Market</td><td style="padding:4px 0;font-weight:bold;">{pick['market']}</td></tr>
        <tr><td style="padding:4px 0;color:#777;">Selection</td><td style="padding:4px 0;font-weight:bold;">{pick['selection']}</td></tr>
        <tr><td style="padding:4px 0;color:#777;">Odds</td><td style="padding:4px 0;">{pick['odds']:.2f} ({pick.get('bookmaker') or 'best available'})</td></tr>
        <tr><td style="padding:4px 0;color:#777;">Estimated probability</td><td style="padding:4px 0;">{pick['model_prob']*100:.1f}%</td></tr>
        <tr><td style="padding:4px 0;color:#777;">Edge vs market</td><td style="padding:4px 0;">{edge_pct}</td></tr>
        <tr><td style="padding:4px 0;color:#777;">Expected value</td><td style="padding:4px 0;color:#0a7d32;font-weight:bold;">{ev_pct}</td></tr>
      </table>
      <div style="margin-top:10px;font-size:13px;color:#444;font-style:italic;">
        {pick.get('justification', '')}
      </div>
    </div>
    """


def _format_pick_text(rank: int, pick: dict) -> str:
    edge_pct = f"{pick['edge']*100:.1f}%" if pick.get("edge") is not None else "n/a"
    return (
        f"Pick #{rank}: {pick['home_team']} vs {pick['away_team']} ({pick['league']})\n"
        f"  Market: {pick['market']}\n"
        f"  Selection: {pick['selection']}\n"
        f"  Odds: {pick['odds']:.2f} ({pick.get('bookmaker') or 'best available'})\n"
        f"  Estimated probability: {pick['model_prob']*100:.1f}%\n"
        f"  Edge vs market: {edge_pct}\n"
        f"  Expected value: {pick['expected_value']*100:.1f}%\n"
        f"  {pick.get('justification', '')}\n"
    )


def compose_email(picks: list, run_time: dt.datetime, matches_seen: int) -> tuple:
    """Returns (subject, plain_text_body, html_body)."""
    date_str = run_time.strftime("%A, %d %B %Y")
    time_str = run_time.strftime("%H:%M UTC")

    if picks:
        subject = f"Value Bet Finder - {len(picks)} pick(s) for {run_time.strftime('%Y-%m-%d')}"
    else:
        subject = f"Value Bet Finder - No qualifying picks for {run_time.strftime('%Y-%m-%d')}"

    text_parts = [
        f"Value Bet Finder\nAnalysis run: {date_str} at {time_str}\n"
        f"Matches analysed: {matches_seen}\n\n"
    ]
    html_parts = [f"""
    <div style="font-family:Arial,Helvetica,sans-serif;max-width:640px;margin:0 auto;">
      <h2 style="color:#1a1a1a;">Value Bet Finder</h2>
      <p style="color:#555;">Analysis run: <b>{date_str} at {time_str}</b><br>
         Matches analysed: <b>{matches_seen}</b></p>
    """]

    if not picks:
        msg = "No bets met the minimum value/confidence threshold today. No picks to report."
        text_parts.append(msg + "\n")
        html_parts.append(f"<p style='color:#a00;font-weight:bold;'>{msg}</p>")
    else:
        for i, pick in enumerate(picks, start=1):
            text_parts.append(_format_pick_text(i, pick))
            html_parts.append(_format_pick_html(i, pick))

    footer_text = (
        "\n--\nThis is an automated analysis for informational purposes only. "
        "It is not financial advice. Bet responsibly.\n"
    )
    footer_html = (
        "<p style='font-size:11px;color:#999;margin-top:20px;'>"
        "This is an automated analysis for informational purposes only. "
        "It is not financial advice. Bet responsibly.</p></div>"
    )
    text_parts.append(footer_text)
    html_parts.append(footer_html)

    return subject, "".join(text_parts), "".join(html_parts)


def send_email(subject: str, text_body: str, html_body: str,
                to_addr: str = None, max_retries: int = None) -> bool:
    """Send the composed email via Gmail SMTP, retrying with exponential
    backoff. Returns True on success, False if all attempts failed."""
    to_addr = to_addr or config.EMAIL_TO
    max_retries = max_retries or config.EMAIL_MAX_RETRIES

    if not config.EMAIL_USER or not config.EMAIL_PASSWORD:
        logger.error("EMAIL_USER / EMAIL_PASSWORD not configured; cannot send email")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_USER
    msg["To"] = to_addr
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
                server.sendmail(config.EMAIL_USER, [to_addr], msg.as_string())
            logger.info("Email sent successfully to %s (attempt %d)", to_addr, attempt)
            return True
        except Exception as exc:
            logger.error("Email send attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                sleep_for = config.EMAIL_RETRY_BACKOFF_SECONDS * attempt
                logger.info("Retrying email send in %d seconds...", sleep_for)
                time.sleep(sleep_for)

    logger.error("All %d email send attempts failed. Giving up.", max_retries)
    return False
