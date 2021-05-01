from sid.plotting import plot_policy_gantt_chart


POLICIES_FOR_GANTT_CHART = {
    "closed_schools": {
        "affected_contact_model": "school",
        "start": "2020-03-09",
        "end": "2020-05-30",
    },
    "partially_closed_schools": {
        "affected_contact_model": "school",
        "start": "2020-06-01",
        "end": "2020-09-30",
    },
    "partially_closed_kindergarden": {
        "affected_contact_model": "school",
        "start": "2020-05-20",
        "end": "2020-06-30",
    },
    "work_closed": {
        "affected_contact_model": "work",
        "start": "2020-03-09",
        "end": "2020-06-15",
    },
    "work_partially_opened": {
        "affected_contact_model": "work",
        "start": "2020-05-01",
        "end": "2020-08-15",
    },
    "closed_leisure_activities": {
        "affected_contact_model": "leisure",
        "start": "2020-03-09",
    },
}


def test_plot_policy_gantt_chart():
    fig, ax = plot_policy_gantt_chart(POLICIES_FOR_GANTT_CHART)
