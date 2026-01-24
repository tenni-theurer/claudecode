#!/usr/bin/env python3
"""
Aughthour Mileage Tracker

Pulls calendar events and calculates mileage from home address for expense reporting.

Usage:
  python mileage_tracker.py                     # Previous month (for cron)
  python mileage_tracker.py 2025               # Specific year
  python mileage_tracker.py 2025-01            # Specific month
  python mileage_tracker.py 2025-01 2025-03    # Date range (YYYY-MM)
  python mileage_tracker.py --no-email         # Skip sending email
"""

import csv
import os
import pickle
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path

import googlemaps
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Configuration
HOME_ADDRESS = "1112 Blackfield Way, Mountain View, CA 94040"
OUTPUT_DIR = Path(__file__).parent / "output"
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"
TOKEN_FILE = Path(__file__).parent / "token.pickle"

# Google Calendar API scope
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Email Configuration
EMAIL_TO = "tenni.theurer@gmail.com"
EMAIL_FROM = "tenni.theurer@gmail.com"


def get_gmail_app_password():
    """Get Gmail App Password from environment or .env file."""
    password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not password:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("GMAIL_APP_PASSWORD="):
                        password = line.strip().split("=", 1)[1]
                        break
    return password


def send_email_with_attachment(subject: str, body: str, attachment_path: Path):
    """Send email via Gmail SMTP with CSV attachment."""
    password = get_gmail_app_password()
    if not password:
        print("  WARNING: GMAIL_APP_PASSWORD not set. Email not sent.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        # Attach CSV file
        if attachment_path.exists():
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={attachment_path.name}"
            )
            msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("  Email sent successfully!")
        return True
    except Exception as e:
        print(f"  WARNING: Email failed: {e}")
        return False


def get_calendar_service():
    """Authenticate and return Google Calendar service."""
    creds = None

    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print(f"ERROR: {CREDENTIALS_FILE} not found.")
                print("\nTo set up Google Calendar API:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create a new project (or select existing)")
                print("3. Enable 'Google Calendar API'")
                print("4. Go to Credentials > Create Credentials > OAuth client ID")
                print("5. Select 'Desktop app' as application type")
                print("6. Download the JSON and save as 'credentials.json' in this folder")
                sys.exit(1)

            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    return build("calendar", "v3", credentials=creds)


def get_maps_client():
    """Return Google Maps client for distance calculation."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("GOOGLE_MAPS_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break

    if not api_key:
        print("WARNING: GOOGLE_MAPS_API_KEY not set. Mileage will not be calculated.")
        print("Add it to .env file or set as environment variable.")
        return None

    return googlemaps.Client(key=api_key)


def calculate_miles(maps_client, destination):
    """Calculate driving miles from home to destination."""
    if not maps_client or not destination:
        return None

    try:
        result = maps_client.distance_matrix(
            origins=[HOME_ADDRESS],
            destinations=[destination],
            mode="driving",
            units="imperial"
        )

        if result["rows"][0]["elements"][0]["status"] == "OK":
            # Distance is returned in meters, convert to miles
            meters = result["rows"][0]["elements"][0]["distance"]["value"]
            miles = meters / 1609.344
            return round(miles, 1)
    except Exception as e:
        print(f"  Warning: Could not calculate distance to '{destination}': {e}")

    return None


def get_events(service, start_date, end_date):
    """Fetch calendar events within date range."""
    events_result = service.events().list(
        calendarId="primary",
        timeMin=start_date.isoformat() + "Z",
        timeMax=end_date.isoformat() + "Z",
        singleEvents=True,
        orderBy="startTime",
        maxResults=2500
    ).execute()

    return events_result.get("items", [])


# Keywords that indicate personal (non-work) events
PERSONAL_KEYWORDS = [
    "pilates", "yoga", "gym", "workout", "fitness", "doctor", "dentist",
    "haircut", "salon", "spa", "massage", "personal", "family", "birthday",
    "wedding", "party", "dinner with", "brunch with", "vacation", "flight to",
    "hotel", "airbnb"
]

# Locations that indicate personal (non-work) events
PERSONAL_LOCATIONS = [
    "club pilates", "pilates", "yoga", "gym", "fitness", "24 hour fitness",
    "equinox", "orangetheory", "crossfit"
]

# Keywords that indicate work-related events
WORK_KEYWORDS = [
    "meeting", "sync", "coffee", "lunch", "f2f", "work", "standup", "review",
    "interview", "call", "demo", "presentation", "conference", "summit",
    "hackathon", "workshop", "networking", "investor", "founder", "ceo",
    "cto", "cpo", "vp", "director", "council", "fellowship", "cohort",
    "ai", "startup", "venture", "pitch", "board", "dinner", "reception",
    "alumni", "visa"
]

# Keywords that indicate NOT work (override other signals)
NOT_WORK_KEYWORDS = [
    "jury duty", "block", "pilates", "yoga", "gym", "doctor", "dentist"
]


def is_personal_event(event):
    """Check if event appears to be personal (non-work)."""
    title = event.get("summary", "").lower()
    description = event.get("description", "").lower()
    location = event.get("location", "").lower()

    # Check for personal keywords in title/description
    for keyword in PERSONAL_KEYWORDS:
        if keyword in title or keyword in description:
            return True

    # Check for personal locations
    for loc_keyword in PERSONAL_LOCATIONS:
        if loc_keyword in location:
            return True

    return False


def looks_like_meeting_with_person(title):
    """Check if title looks like a 1:1 or meeting with a person."""
    title_lower = title.lower()

    # Patterns that indicate meeting with a person:
    # "Name and Name", "Name / Name", "Name <> Name", "Name x Name"
    # "Coffee with Name", "Lunch w/ Name", "Meet Name"

    # Contains "and" between names (e.g., "Tenni and Marty")
    if " and " in title_lower:
        return True

    # Contains "/" or "<>" separator (e.g., "Tenni / Marty", "Tenni<>Marty")
    if " / " in title or "/" in title or "<>" in title or " x " in title_lower:
        return True

    # Contains common meeting prefixes
    meeting_prefixes = ["coffee with", "lunch with", "lunch w/", "meet with",
                        "meeting with", "chat with", "call with", "sync with"]
    for prefix in meeting_prefixes:
        if prefix in title_lower:
            return True

    # Known colleague names (single word titles that are people)
    known_colleagues = ["marty", "sunil", "jay", "tom"]
    if title_lower.strip() in known_colleagues:
        return True

    # Short title (1-2 words) that starts with capital letter is likely a person's name
    words = title.strip().split()
    if len(words) <= 2 and title[0].isupper() and title_lower not in NOT_WORK_KEYWORDS:
        # Likely a person's name
        return True

    return False


def is_likely_work(event):
    """Check if event is likely work-related."""
    title = event.get("summary", "")
    title_lower = title.lower()
    description = event.get("description", "").lower()
    attendees = event.get("attendees", [])

    # Check for explicit NOT work keywords first
    for keyword in NOT_WORK_KEYWORDS:
        if keyword in title_lower:
            return False

    # Has work keywords
    for keyword in WORK_KEYWORDS:
        if keyword in title_lower or keyword in description:
            return True

    # Looks like a 1:1 or meeting with a person
    if looks_like_meeting_with_person(title):
        return True

    # Has attendees (likely a meeting)
    if len(attendees) >= 1:
        return True

    return False


def filter_meeting_events(events):
    """Filter events that have locations, excluding personal events."""
    meeting_events = []

    for event in events:
        location = event.get("location", "").strip()

        # Skip events without locations
        if not location:
            continue

        # Skip personal events (like Pilates)
        if is_personal_event(event):
            continue

        meeting_events.append(event)

    return meeting_events


def format_event_for_csv(event, miles):
    """Format a calendar event for CSV output."""
    # Parse start/end times
    start = event.get("start", {})
    end = event.get("end", {})

    if "dateTime" in start:
        start_dt = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end["dateTime"].replace("Z", "+00:00"))
        date_str = start_dt.strftime("%m/%d/%Y")
        time_from = start_dt.strftime("%I:%M %p")
        time_to = end_dt.strftime("%I:%M %p")
    else:
        # All-day event
        date_str = datetime.fromisoformat(start["date"]).strftime("%m/%d/%Y")
        time_from = "All day"
        time_to = "All day"

    # Get attendees
    attendees = event.get("attendees", [])
    attendee_emails = [a.get("email", "") for a in attendees if not a.get("self", False)]
    attendees_str = "; ".join(attendee_emails)

    # Flag if likely work-related
    likely_work = is_likely_work(event)
    work_flag = "Yes" if likely_work else "Review"

    return {
        "date": date_str,
        "time_from": time_from,
        "time_to": time_to,
        "title": event.get("summary", "No title"),
        "attendees": attendees_str,
        "description": event.get("description", "")[:500],  # Truncate long descriptions
        "location": event.get("location", ""),
        "miles_from_home": miles if miles else "",
        "round_trip_miles": miles * 2 if miles else "",
        "work_related": work_flag
    }


def main():
    """Main function."""
    # Parse command line arguments
    now = datetime.now(timezone.utc)
    send_email = "--no-email" not in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-email"]

    if len(args) == 0:
        # Default: previous month (for cron automation)
        if now.month == 1:
            start_date = datetime(now.year - 1, 12, 1)
            end_date = datetime(now.year, 1, 1) - timedelta(seconds=1)
        else:
            start_date = datetime(now.year, now.month - 1, 1)
            end_date = datetime(now.year, now.month, 1) - timedelta(seconds=1)
    elif len(args) == 1:
        arg = args[0]
        if "-" in arg:
            # Specific month: YYYY-MM
            parts = arg.split("-")
            year, month = int(parts[0]), int(parts[1])
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        else:
            # Specific year
            year = int(arg)
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31, 23, 59, 59)
    elif len(args) == 2:
        # Date range (YYYY-MM format)
        start_parts = args[0].split("-")
        end_parts = args[1].split("-")
        start_date = datetime(int(start_parts[0]), int(start_parts[1]), 1)
        # End of the specified month
        end_year, end_month = int(end_parts[0]), int(end_parts[1])
        if end_month == 12:
            end_date = datetime(end_year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end_date = datetime(end_year, end_month + 1, 1) - timedelta(seconds=1)
    else:
        print("Usage: python mileage_tracker.py [YYYY | YYYY-MM | YYYY-MM YYYY-MM] [--no-email]")
        sys.exit(1)

    print(f"Aughthour Mileage Tracker")
    print(f"=" * 50)
    print(f"Home address: {HOME_ADDRESS}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()

    # Connect to Google Calendar
    print("Connecting to Google Calendar...")
    service = get_calendar_service()

    # Get Maps client for mileage
    maps_client = get_maps_client()

    # Fetch events
    print("Fetching calendar events...")
    all_events = get_events(service, start_date, end_date)
    print(f"  Found {len(all_events)} total events")

    # Filter for in-person meetings
    meeting_events = filter_meeting_events(all_events)
    print(f"  Found {len(meeting_events)} events with locations")
    print()

    # Process events and calculate mileage
    print("Processing events and calculating mileage...")
    rows = []
    total_miles = 0

    for event in meeting_events:
        location = event.get("location", "")
        miles = calculate_miles(maps_client, location)

        if miles:
            total_miles += miles * 2  # Round trip

        row = format_event_for_csv(event, miles)
        rows.append(row)
        print(f"  {row['date']} - {row['title'][:40]}: {miles or 'N/A'} miles")

    # Output to CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"mileage_{start_date.strftime('%Y%m')}_to_{end_date.strftime('%Y%m')}.csv"

    with open(output_file, "w", newline="") as f:
        fieldnames = ["date", "time_from", "time_to", "title", "attendees", "description", "location", "miles_from_home", "round_trip_miles", "work_related"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Count events with actual addresses (where miles were calculated)
    events_with_miles = [r for r in rows if r["miles_from_home"]]

    # Count events by month
    events_by_month = {}
    events_with_miles_by_month = {}
    for row in rows:
        month = row["date"][:2]  # MM from MM/DD/YYYY
        events_by_month[month] = events_by_month.get(month, 0) + 1
        if row["miles_from_home"]:
            events_with_miles_by_month[month] = events_with_miles_by_month.get(month, 0) + 1

    print()
    print(f"=" * 50)
    print(f"Summary:")
    print(f"  Total events with locations: {len(rows)}")
    print(f"  Events with real addresses: {len(events_with_miles)}")
    print(f"  Total round-trip miles: {total_miles:.1f}")
    print()
    print(f"Events by month:")
    for month in sorted(events_by_month.keys()):
        month_name = datetime(2025, int(month), 1).strftime("%B")
        with_miles = events_with_miles_by_month.get(month, 0)
        print(f"  {month_name}: {events_by_month[month]} events ({with_miles} with mileage)")
    print()
    print(f"Output saved to: {output_file}")

    # Send email with summary and attachment
    if send_email:
        month_range = f"{start_date.strftime('%B %Y')}"
        if start_date.month != end_date.month or start_date.year != end_date.year:
            month_range = f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}"

        subject = f"Aughthour Mileage Report: {month_range}"

        body = f"""Hi Tenni,

Here's your mileage report for {month_range}.

SUMMARY
{'=' * 40}
Total events with locations: {len(rows)}
Events with real addresses: {len(events_with_miles)}
Total round-trip miles: {total_miles:.1f}

EVENTS BY MONTH
{'=' * 40}
"""
        for month in sorted(events_by_month.keys()):
            month_name = datetime(2025, int(month), 1).strftime("%B")
            with_miles = events_with_miles_by_month.get(month, 0)
            body += f"{month_name}: {events_by_month[month]} events ({with_miles} with mileage)\n"

        body += f"""

REMINDER: Submit your expenses to Mercury!
https://app.mercury.com/

The CSV file is attached for your records.

- Aughthour Mileage Tracker
"""

        send_email_with_attachment(subject, body, output_file)


if __name__ == "__main__":
    main()
