import unittest
from datetime import datetime, timezone

from forex_factory_calendar import CalendarFetchError, parse_calendar_html


class ForexFactoryCalendarTests(unittest.TestCase):
    def test_parses_event_fields_and_utc_timestamp(self):
        html = """
        <table>
          <tr class="calendar__row">
            <td class="calendar__date">Mon Oct 12</td>
            <td class="calendar__time">8:30am</td>
            <td class="calendar__currency">USD</td>
            <td class="calendar__impact"><span class="ff-impact-red" title="High Impact Expected"></span></td>
            <td class="calendar__event"><span class="calendar__event-title">CPI m/m</span></td>
            <td class="calendar__actual">0.3%</td>
            <td class="calendar__forecast">0.2%</td>
            <td class="calendar__previous">0.4%</td>
          </tr>
        </table>
        """
        now = datetime(2026, 10, 12, 0, 0, tzinfo=timezone.utc)
        events = parse_calendar_html(html, now=now)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["currency"], "USD")
        self.assertEqual(events[0]["impact"], "High")
        self.assertEqual(events[0]["event"], "CPI m/m")
        self.assertEqual(events[0]["datetime_utc"], "2026-10-12T08:30:00+00:00")
        self.assertEqual(events[0]["actual"], "0.3%")
        self.assertEqual(events[0]["forecast"], "0.2%")
        self.assertEqual(events[0]["previous"], "0.4%")

    def test_rejects_unrecognized_calendar_markup(self):
        with self.assertRaises(CalendarFetchError):
            parse_calendar_html("<html><body>Access denied</body></html>")


if __name__ == "__main__":
    unittest.main()
