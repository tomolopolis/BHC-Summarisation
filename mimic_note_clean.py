import re

import numpy as np
import pandas as pd

from section_parser import section_radiology_text, section_echo_text, section_nurs_notes, section_physician_prog_text, \
    section_physician_intensivist_note, section_physician_attnd_prog_text, \
    section_resp_care_shft_note

"""
Adapted from: https://github.com/abachaa/MEDIQA2021/tree/main/Task3 for MIMIC-III
"""


def _pull_out_section(df: pd.DataFrame, report_section_func: callable, section_name='impression',
                      concat_sections=True, selected_sec_idx=0):
    def _pull_out_impression_section(row: pd.Series):
        sections, section_names, section_idx = report_section_func(row.text)
        if concat_sections:
            section = '\n'.join([sections[i] for i in np.argwhere(np.array(section_names) == section_name).flatten()])
        elif selected_sec_idx != 0:
            idxs = np.argwhere(np.array(section_names) == section_name).flatten()
            if selected_sec_idx >= len(idxs) or len(idxs) == 0:
                section = ''
            else:
                section = sections[idxs[selected_sec_idx]]
        elif section_name in section_names:
            section = sections[section_names.index(section_name)]
        else:
            # do nothing here, but potentially also use findings or other sections...
            section = None
        return section, sections, section_names

    return df.apply(_pull_out_impression_section, axis=1)


def parse_phys_prog_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_prog_text, 'assessment and plan', concat_sections=False)


def parse_phys_intens_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_intensivist_note, 'assessment and plan', concat_sections=False,
                             selected_sec_idx=1)


def parse_phys_attend_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_attnd_prog_text, 'assessment and plan', concat_sections=False,
                             selected_sec_idx=-1)


def parse_phys_res_attnd_adm_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_prog_text, 'assessment and plan', concat_sections=False)


def parse_phys_res_attn_micu_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_prog_text, 'assessment and plan', concat_sections=False,
                             selected_sec_idx=-1)


def parse_phys_res_attn_prog_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_prog_text, 'assessment and plan', concat_sections=False,
                             selected_sec_idx=0)


def parse_phys_surgical_adm_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_physician_prog_text, 'assessment and plan', concat_sections=False,
                             selected_sec_idx=-1)


def parse_respiratory_care_shift_note(df: pd.DataFrame):
    return _pull_out_section(df, section_resp_care_shft_note, 'plan')


def parse_nurs_prog_notes_text(df: pd.DataFrame):
    return _pull_out_section(df, section_nurs_notes, 'assessment')


def parse_echo_report_text(df: pd.DataFrame):
    return _pull_out_section(df, section_echo_text)


def parse_radiology_report_text(df: pd.DataFrame):
    return _pull_out_section(df, section_radiology_text)


def clean_findings(text):
    """ Clean up the findings string.
    """
    text = text.strip().replace('\n', '')
    # reduce consecutive spaces
    text = re.sub(r'\s\s+', ' ', text)
    return text


def clean_impression(text):
    """ Clean up the impression string.
    This mainly removes bullet numbers for consistency.
    """
    text = text.strip().replace('\n', '')
    # remove bullet numbers
    text = re.sub(r'^[0-9]\.\s+', '', text)
    text = re.sub(r'\s[0-9]\.\s+', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    return text


def clean_background(text):
    """ Clean up the background string.
    """
    text = text.strip()
    # remove common prefix title
    if text.startswith('FINAL REPORT'):
        text = text[12:].lstrip()
    # remove findings header
    if text.endswith(':'):
        idx = text.rfind(' ')
        text = text[:idx].rstrip()
    text = re.sub(r'\s\s+', ' ', text)
    text = text.strip().replace('\n', '')
    return text


if __name__ == '__main__':
    # nurse_prog_reports = pd.read_csv('nurs_prog_notes_sample.csv')
    # parsed_reports = parse_nurs_prog_notes_text(nurse_prog_reports)
    # note_sample = pd.read_csv('echo_report_sample.csv')
    # impressions = parse_echo_report_text(note_sample)
    # note_sample = pd.read_csv('no_impression_radio_reports.csv')
    # impressions = parse_radiology_report_text(note_sample)
    # sections = parse_phys_intens_notes_text(pd.read_csv('intensivist_note_sample.csv'))
    sections = parse_phys_attend_notes_text(pd.read_csv('phys_attnd_prog_notes.csv'))
    print(len(sections))
