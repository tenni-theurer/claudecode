#!/usr/bin/env python3
"""
University Tour Availability Checker
Checks multiple university tour pages for available dates

Usage:
  python check_tours.py                      # Check current month
  python check_tours.py April                # Check April (all dates)
  python check_tours.py April 6-10           # Check April 6-10 only
  python check_tours.py "April 2026"         # Check April 2026
  python check_tours.py "April 2026" 6-10    # Check April 6-10, 2026
"""

import asyncio
import json
import os
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration
RESULTS_FILE = Path(__file__).parent / "results.json"
LOG_FILE = Path(__file__).parent / "checker.log"
SUMMARY_FILE = Path(__file__).parent / "summary.txt"

# Email Configuration
EMAIL_TO = "tenni.theurer@gmail.com"
EMAIL_FROM = "tenni.theurer@gmail.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

UNIVERSITIES = [
    {
        "name": "NYU",
        "url": "https://connect.nyu.edu/portal/nyuvisit_tours",
    },
    {
        "name": "UPenn",
        "url": "https://key.admissions.upenn.edu/portal/visit-connect?tab=campus-visits",
    },
    {
        "name": "Columbia",
        "url": "https://admissions.columbiasc.edu/portal/campus_visit",
    },
    {
        "name": "UMichigan",
        "url": "https://enrollmentconnect.umich.edu/portal/campus_tours",
    },
    {
        "name": "Rutgers",
        "url": "https://events.blackthorn.io/en/1UyCAW6/g/RutgersNB?keywords=Bus%20Tour&date=2026-04-05,2026-04-12",
    },
    {
        "name": "BostonU",
        "url": "https://www.bu.edu/admissions/visit-us/events/events-calendar/?date=2026-04-08#calendar",
    },
    {
        "name": "BostonCollege",
        "url": "https://admission.bc.edu/portal/campusvisit",
    },
    {
        "name": "Northeastern",
        "url": "https://apply.northeastern.edu/portal/campus-visit?tab=Visit_our_Boston_Campus",
    },
]


def parse_args():
    """Parse command line arguments"""
    # Defaults
    now = datetime.now()
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    target_month = month_names[now.month - 1]
    target_year = str(now.year)
    target_dates = None  # None means all dates

    if len(sys.argv) > 1:
        # First arg: month (and optionally year)
        month_arg = sys.argv[1]

        # Check if year is included (e.g., "April 2026")
        parts = month_arg.split()
        if len(parts) == 2 and parts[1].isdigit():
            target_month = parts[0]
            target_year = parts[1]
        else:
            target_month = month_arg
            # If month is before current month, assume next year
            if target_month in month_names:
                month_idx = month_names.index(target_month)
                if month_idx < now.month - 1:
                    target_year = str(now.year + 1)

    if len(sys.argv) > 2:
        # Second arg: date range (e.g., "6-10" or "6,7,8,9,10")
        date_arg = sys.argv[2]
        if "-" in date_arg:
            start, end = date_arg.split("-")
            target_dates = list(range(int(start), int(end) + 1))
        elif "," in date_arg:
            target_dates = [int(d.strip()) for d in date_arg.split(",")]
        else:
            target_dates = [int(date_arg)]

    return target_month, target_year, target_dates


def log(message: str):
    """Log message to file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def send_notification(title: str, message: str):
    """Send macOS notification"""
    os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')


def send_email(subject: str, body: str):
    """Send email via Gmail SMTP"""
    if not GMAIL_APP_PASSWORD:
        log("  ⚠️  Email not sent: GMAIL_APP_PASSWORD not set")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, GMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        log("  ✉️  Email sent successfully")
        return True
    except Exception as e:
        log(f"  ⚠️  Email failed: {e}")
        return False


def get_months_to_capture(target_month: str, target_year: str) -> list:
    """Get list of months from January 2026 to target month"""
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    start_year = 2026
    start_month_idx = 0  # January

    target_year_int = int(target_year)
    target_month_idx = month_names.index(target_month)

    months = []
    year = start_year
    month_idx = start_month_idx

    while True:
        months.append((month_names[month_idx], year))
        if year == target_year_int and month_idx == target_month_idx:
            break
        month_idx += 1
        if month_idx > 11:
            month_idx = 0
            year += 1
        # Safety limit
        if len(months) > 24:
            break

    return months


async def navigate_to_month_with_screenshots(page, target_month: str, target_year: str, university_name: str, screenshot_dir: Path):
    """Navigate to target month, taking screenshots of each month along the way"""
    months_to_capture = get_months_to_capture(target_month, target_year)

    next_buttons = [
        "button:has-text('>')",
        "button:has-text('Next')",
        "button:has-text('›')",
        "[aria-label*='next']",
        "[aria-label*='Next']",
        "[class*='next']",
        ".fc-next-button",
        "[data-action='next']",
    ]

    captured_months = []

    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    for month_name, year in months_to_capture:
        # Try to reach this month
        for attempt in range(12):
            page_text = await page.evaluate("() => document.body.innerText")
            if month_name in page_text and str(year) in page_text:
                # Found the month, take screenshot
                await page.wait_for_timeout(500)
                month_num = str(month_names.index(month_name) + 1).zfill(2)
                screenshot_name = f"{university_name.lower()}_{month_num}_{year}.png"
                await page.screenshot(path=str(screenshot_dir / screenshot_name), full_page=True)
                log(f"    Screenshot: {month_name} {year}")
                captured_months.append(f"{month_name} {year}")
                break

            # Click next to advance
            clicked = False
            for selector in next_buttons:
                try:
                    btn = await page.query_selector(selector)
                    if btn:
                        await btn.click()
                        await page.wait_for_timeout(500)
                        clicked = True
                        break
                except:
                    pass

            if not clicked:
                break

    # Check if we reached the target
    page_text = await page.evaluate("() => document.body.innerText")
    reached = target_month in page_text

    return reached, captured_months


async def navigate_to_month(page, target_month: str):
    """Try to navigate calendar to target month (legacy function for compatibility)"""
    next_buttons = [
        "button:has-text('>')",
        "button:has-text('Next')",
        "button:has-text('›')",
        "[aria-label*='next']",
        "[aria-label*='Next']",
        "[class*='next']",
        ".fc-next-button",
        "[data-action='next']",
    ]

    for attempt in range(6):
        page_text = await page.evaluate("() => document.body.innerText")
        if target_month in page_text:
            log(f"    Reached {target_month}")
            return True

        clicked = False
        for selector in next_buttons:
            try:
                btn = await page.query_selector(selector)
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(500)
                    clicked = True
                    break
            except:
                pass

        if not clicked:
            break

    return False


async def find_available_dates_bu(page, target_dates: list = None) -> list:
    """
    Special handler for Boston University calendar.
    Red dates = available events
    Black dates = at capacity
    Grey dates = no events
    """
    available = set()

    try:
        # BU calendar uses table cells with links for available dates
        # Find all cells in the calendar that contain links (red/available dates)
        cells = await page.query_selector_all("td")

        for cell in cells:
            # Check if cell contains a link (indicates available date)
            link = await cell.query_selector("a")
            if not link:
                continue

            text = (await link.inner_text()).strip()
            if not text.isdigit():
                continue

            day = int(text)
            if day < 1 or day > 31:
                continue

            if target_dates and day not in target_dates:
                continue

            # Get the color of the link to check if it's red (available) vs black (at capacity)
            color = await link.evaluate("""el => {
                const style = window.getComputedStyle(el);
                return style.color;
            }""")

            # Red color indicates available (rgb(204, 0, 0) or similar red values)
            # Black/dark color indicates at capacity
            is_red = False
            if color:
                # Check for red-ish colors (high red, low green/blue)
                if "rgb" in color:
                    # Parse rgb(r, g, b)
                    parts = color.replace("rgb(", "").replace(")", "").split(",")
                    if len(parts) >= 3:
                        r, g, b = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                        # Red if red channel is high and green/blue are low
                        is_red = r > 150 and g < 100 and b < 100

            if is_red:
                available.add(day)

    except Exception as e:
        pass

    return sorted(list(available))


async def find_available_dates_strict(page, university_name: str, target_dates: list = None) -> list:
    """
    Find available dates - must have explicit 'available' class
    and NOT have unavailable/filled/disabled indicators

    If target_dates is provided, only return dates in that list
    """
    # Use special handler for Boston University
    if university_name == "BostonU":
        return await find_available_dates_bu(page, target_dates)

    available = set()

    available_selectors = [
        "td.available",
        "td.open",
        ".day.available",
        "[class*='available']:not([class*='unavailable'])",
        "td.has-event:not(.unavailable):not(.filled)",
    ]

    unavailable_indicators = [
        "unavailable", "filled", "disabled", "closed", "full",
        "booked", "soldout", "sold-out", "inactive", "past"
    ]

    for selector in available_selectors:
        try:
            elements = await page.query_selector_all(selector)
            for el in elements:
                text = (await el.inner_text()).strip()

                if text.isdigit():
                    day = int(text)
                    if 1 <= day <= 31:
                        # If we have target dates, skip if not in list
                        if target_dates and day not in target_dates:
                            continue

                        class_attr = (await el.get_attribute("class") or "").lower()

                        try:
                            parent_class = await el.evaluate("el => el.parentElement ? el.parentElement.className : ''")
                            parent_class = parent_class.lower()
                        except:
                            parent_class = ""

                        all_classes = class_attr + " " + parent_class

                        has_available = "available" in all_classes or "open" in all_classes
                        has_unavailable = any(ind in all_classes for ind in unavailable_indicators)

                        if has_available and not has_unavailable:
                            available.add(day)
        except:
            pass

    # Also check table cells directly
    try:
        cells = await page.query_selector_all("td")
        for cell in cells:
            text = (await cell.inner_text()).strip()
            if text.isdigit():
                day = int(text)
                if 1 <= day <= 31:
                    if target_dates and day not in target_dates:
                        continue

                    class_attr = (await cell.get_attribute("class") or "").lower()

                    if "available" in class_attr and not any(ind in class_attr for ind in unavailable_indicators):
                        available.add(day)
    except:
        pass

    return sorted(list(available))


async def check_university(page, university: dict, target_month: str, target_year: str, target_dates: list) -> dict:
    """Check a single university for available tour dates"""
    name = university["name"]
    url = university["url"]

    log(f"Checking {name}...")
    result = {
        "name": name,
        "url": url,
        "checked_at": datetime.now().isoformat(),
        "target_month": f"{target_month} {target_year}",
        "target_dates": f"{target_dates[0]}-{target_dates[-1]}" if target_dates else "all",
        "available_dates": [],
        "reached_target_month": False,
        "screenshots": [],
        "error": None,
    }

    try:
        await page.goto(url, timeout=45000)
        await page.wait_for_timeout(3000)

        screenshot_dir = Path(__file__).parent / "screenshots"
        screenshot_dir.mkdir(exist_ok=True)

        # Navigate and capture screenshots of each month from Jan 2026 to target
        reached, captured_months = await navigate_to_month_with_screenshots(
            page, target_month, target_year, name, screenshot_dir
        )
        result["reached_target_month"] = reached
        result["screenshots"] = captured_months

        if reached:
            await page.wait_for_timeout(1000)

            available = await find_available_dates_strict(page, name, target_dates)
            result["available_dates"] = [f"{target_month} {d}" for d in available]

        if result["available_dates"]:
            dates_str = ", ".join([str(d) for d in sorted([int(x.split()[-1]) for x in result["available_dates"]])])
            log(f"  ✓ {name}: AVAILABLE - {dates_str}")
        elif reached:
            log(f"  ✗ {name}: No available dates found")
        else:
            log(f"  ? {name}: Could not navigate to {target_month}")

    except Exception as e:
        result["error"] = str(e)
        log(f"  ✗ {name}: Error - {e}")
        try:
            screenshot_dir = Path(__file__).parent / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            await page.screenshot(path=str(screenshot_dir / f"{name.lower()}_error.png"))
        except:
            pass

    return result


async def main():
    """Main function to check all universities"""
    target_month, target_year, target_dates = parse_args()

    log("=" * 50)
    log("University Tour Availability Checker")
    log(f"Target: {target_month} {target_year}")
    if target_dates:
        log(f"Dates: {target_dates[0]}-{target_dates[-1]}")
    else:
        log("Dates: All available")
    log("=" * 50)

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        for university in UNIVERSITIES:
            result = await check_university(page, university, target_month, target_year, target_dates)
            results.append(result)

        await browser.close()

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    log("")
    log("=" * 50)
    log("SUMMARY")
    log("=" * 50)

    for result in results:
        if result["available_dates"]:
            dates = sorted([int(x.split()[-1]) for x in result["available_dates"]])
            log(f"  {result['name']}: {dates}")
        elif result["error"]:
            log(f"  {result['name']}: ERROR")
        else:
            log(f"  {result['name']}: No dates available")

    # Notify if any found
    found_any = any(r["available_dates"] for r in results)
    if found_any:
        summary = [f"{r['name']}: {len(r['available_dates'])}" for r in results if r["available_dates"]]
        send_notification(f"🎓 {target_month} Tours!", "; ".join(summary))

    # Write readable summary file
    with open(SUMMARY_FILE, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("UNIVERSITY TOUR AVAILABILITY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checked: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
        f.write(f"Target: {target_month} {target_year}")
        if target_dates:
            f.write(f" (dates {target_dates[0]}-{target_dates[-1]})")
        f.write("\n")
        f.write("=" * 60 + "\n\n")

        # Group results: available, no dates, could not reach
        available = [r for r in results if r["available_dates"]]
        no_dates = [r for r in results if not r["available_dates"] and r["reached_target_month"] and not r["error"]]
        not_reached = [r for r in results if not r["reached_target_month"] or r["error"]]

        if available:
            f.write("✅ AVAILABLE\n")
            f.write("-" * 40 + "\n")
            for result in available:
                dates = sorted([int(x.split()[-1]) for x in result["available_dates"]])
                f.write(f"  {result['name']}: {', '.join(str(d) for d in dates)}\n")
                f.write(f"    {result['url']}\n\n")

        if no_dates:
            f.write("❌ NO AVAILABLE DATES\n")
            f.write("-" * 40 + "\n")
            for result in no_dates:
                f.write(f"  {result['name']}\n")
                f.write(f"    {result['url']}\n\n")

        if not_reached:
            f.write("⚠️  COULD NOT CHECK\n")
            f.write("-" * 40 + "\n")
            for result in not_reached:
                if result["error"]:
                    f.write(f"  {result['name']}: {result['error']}\n")
                else:
                    f.write(f"  {result['name']}: Could not navigate to {target_month}\n")
                f.write(f"    {result['url']}\n\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Screenshots saved in: screenshots/\n")
        f.write("Full results in: results.json\n")

    log("")
    log("Results saved to results.json")
    log("Summary saved to summary.txt")

    # Send email with summary
    with open(SUMMARY_FILE, "r") as f:
        summary_text = f.read()

    available_schools = [r["name"] for r in results if r["available_dates"]]
    if available_schools:
        subject = f"🎓 Tours Available: {', '.join(available_schools)} - {target_month} {target_year}"
    else:
        subject = f"University Tour Check - {target_month} {target_year} - No Availability"

    send_email(subject, summary_text)

    return results


if __name__ == "__main__":
    asyncio.run(main())
