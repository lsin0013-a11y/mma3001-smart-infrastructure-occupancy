# Calendar inputs

These are public calendar dates, compiled by hand and used as model inputs. Both files are tracked in git.

| File | Contents |
|---|---|
| `monash_teaching_periods.csv` | Monash semester periods (teaching, mid-semester break, SWOT VAC, exams). One row per continuous period; start and end dates are inclusive. Any date not listed is outside semester. |
| `vic_public_holidays.csv` | Victorian statewide public holidays. |

## Sources and caveats

- The official Monash and Business Victoria pages block automated downloads. The dates were taken from those pages as quoted in web search results; the `source` column gives the page for each row.
- The 2025 Monash dates come from the Monash Malaysia principal-dates page. For 2026, that page matches the Australian campus dates.
- The exam period is taken to start on the Saturday after SWOT VAC.
- The date of the 2025 Friday before the AFL Grand Final was not confirmed on the Business Victoria page; it comes from a secondary source.

Check individual dates against the official pages before relying on them.
