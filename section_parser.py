"""
A section parser module that parses a MIMIC-CXR report into multiple sections.

Code adapted from the MEDIQA Task3 shared task:
https://github.com/abachaa/MEDIQA2021/tree/main/Task3
and originally from the MIMIC-CXR GitHub repo at:
https://github.com/MIT-LCP/mimic-cxr/tree/master/txt.
"""

import re

from split import frequent_sections, dis_sum_post_headers, dis_sum_brief_hos_course_headers, \
    dis_sum_pre_headers


def section_nurs_notes(text: str):
    sections = []
    section_names = []

    start = None
    sec_names = ['assessment:', 'action:', 'response:', 'plan:']
    for idx, match in enumerate(re.compile('|'.join(sec_names), re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            sections.append(text[0:match.start()])
        section_names.append(text[match.start():match.end()].lower().strip().replace(':', ''))
        if start:
            sections.append(text[start:match.start()])
        start = match.end()
    if start:
        sections.append(text[start:])
    return sections, section_names, []


def section_phys_texts(text: str, sec_names):
    sections = []
    section_names = []
    curr_sec_text_start = None
    curr_sec_text_end = None

    for idx, group in enumerate(re.compile('|'.join(sec_names), re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            curr_sec_text_start = 0
        sec_name = text[group.start():group.end()].lower().strip().replace(':', '')
        if curr_sec_text_start is not None:
            curr_sec_text_end = group.start() - 1
            sections.append(text[curr_sec_text_start:curr_sec_text_end])
            curr_sec_text_start, curr_sec_text_end = None, None
        section_names.append(sec_name)
        curr_sec_text_start = group.end()
    if curr_sec_text_start:
        sections.append(text[curr_sec_text_start:])
    return sections, section_names, []


def section_discharge_texts(text: str):
    """
    Sections discharge notes into 3 sections. 'Before Hosp_course', 'Hosp_course', and 'After hosp course'
    :param text:
    :return:
    """
    sections = []
    section_names = []
    curr_sec_text_start = None
    curr_sec_text_end = None

    dis_summary_headers = dis_sum_pre_headers + dis_sum_brief_hos_course_headers + dis_sum_post_headers
    headers = '|'.join([f'({n})' for n in dis_summary_headers])

    for idx, group in enumerate(re.compile(headers, re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            curr_sec_text_start = 0
        sec_name = text[group.start():group.end()].lower().strip().replace(':', '')
        if curr_sec_text_start is not None:
            curr_sec_text_end = group.start() - 1
            sections.append(text[curr_sec_text_start:curr_sec_text_end])
            curr_sec_text_start, curr_sec_text_end = None, None
        section_names.append(sec_name)
        curr_sec_text_start = group.end()
    if curr_sec_text_start:
        sections.append(text[curr_sec_text_start:])
    return sections, section_names, []


def section_physician_intensivist_note(text: str):
    return section_phys_texts(text, ['assessment and plan:', 'assessment and plan',
                                     'neurologic:', 'cardiovascular:', 'pulmonary:',
                                     'nutrition:', 'renal:', 'hematology:', 'endocrine:',
                                     'infectious disease:', 'wounds:', 'imaging:', 'fluids:',
                                     'consults:', 'billing diagnosis:', 'icu care'])


def section_physician_prog_text(text: str):
    return section_phys_texts(text, ['assessment and plan', 'nutrition:'])


def section_physician_attnd_prog_text(text: str):
    return section_phys_texts(text, ['cxr:', 'assessment and plan', 'nutrition:'])


def section_resp_care_shft_note(text: str):
    return section_phys_texts(text, ['ventilation assessment', 'plan',
                                     'reason for continuing current ventilatory support:',
                                     'respiratory care shift procedures'])


def section_echo_text(text: str):
    sections = []
    section_names = []
    curr_sec_text_start = None
    curr_sec_text_end = None

    for idx, group in enumerate(re.compile('\n([a-z]+):\n?', re.IGNORECASE).finditer(text)):
        if idx == 0:
            section_names.append('preamble')
            curr_sec_text_start = 0
        sec_name = text[group.start():group.end()].lower().strip().replace(':', '')
        if sec_name in frequent_sections.keys():
            if curr_sec_text_start is not None:
                curr_sec_text_end = group.start() - 1
                sections.append(text[curr_sec_text_start:curr_sec_text_end])
                curr_sec_text_start, curr_sec_text_end = None, None
            section_names.append(frequent_sections[sec_name])
            curr_sec_text_start = group.end()
    if curr_sec_text_start:
        sections.append(text[curr_sec_text_start:])
    return sections, section_names, []


def section_radiology_text(text: str):
    """Splits text into sections.

    Assumes text is in a radiology report format, e.g.:

        COMPARISON:  Chest radiograph dated XYZ.

        IMPRESSION:  ABC...

    Given text like this, it will output text from each section,
    where the section type is determined by the all caps header.

    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(
        r'\n ([A-Z ()/,-]+)\n?[:|;]\s?', re.DOTALL | re.IGNORECASE)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    section_names = normalize_section_names(section_names)

    # remove empty sections
    # this handles when the report starts with a finding-like statement
    #  .. but this statement is not a section, more like a report title
    #  e.g. p10/p10103318/s57408307
    #    CHEST, PA LATERAL:
    #
    #    INDICATION:   This is the actual section ....
    # it also helps when there are multiple findings sections
    # usually one is empty
    for i in reversed(range(len(section_names))):
        if section_names[i] in ('impression', 'findings'):
            if sections[i].strip() == '':
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    if ('impression' not in section_names) & ('findings' not in section_names):
        # create a new section for the final paragraph
        if '\n \n' in sections[-1]:
            sections.append('\n \n'.join(sections[-1].split('\n \n')[1:]))
            sections[-2] = sections[-2].split('\n \n')[0]
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx


def normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    p_findings = [
        'chest',
        'portable',
        'pa and lateral',
        'lateral and pa',
        'ap and lateral',
        'lateral and ap',
        'frontal and',
        'two views',
        'frontal view',
        'pa view',
        'ap view',
        'one view',
        'lateral view',
        'bone window',
        'frontal upright',
        'frontal semi-upright',
        'ribs',
        'pa and lat'
    ]
    p_findings = re.compile('({})'.format('|'.join(p_findings)))

    main_sections = [
        'impression', 'findings', 'history', 'comparison',
        'addendum'
    ]
    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = 'findings'

        # if it looks like it is describing the entire study
        # it's equivalent to findings
        # group similar phrasings for impression

    return section_names


if __name__ == '__main__':
    text = """
Admission Date:  [**2193-4-26**]              Discharge Date:   [**2193-4-29**]

Date of Birth:  [**2114-9-29**]             Sex:   M

Service: MEDICINE

Allergies:
No Known Allergies / Adverse Drug Reactions

Attending:[**First Name3 (LF) 1899**]
Chief Complaint:
transfer for c. cath/STEMI

Major Surgical or Invasive Procedure:
Cardiac catheterization with drug-eluting stent placed in the
left anterior descending artery


History of Present Illness:
79-year-old male with history of CAD and prior PCI with DES to
OM2 at [**Hospital1 2025**] ([**10-7**]) that presented to the ER at OSH with [**2192-2-8**]
chest pain. The night prior to presentation he experienced
indigestion. He then awoke with a "rope-like" non-radiating
chest discomfort with no associated symptoms except perhaps
chills that resolving except the portion "over the heart." He
continued to have this discomfort. His wife called his PCP and
told him to report to the nearest ER. EKG on presentation showed
ST elevation in leads V3,4, and 5. Troponin was 12.483.

He was given 81 mg ASA x 4, 4500 units heparin bolus with drip
at 1800 units/hr and 5 mg IV lopressor. He was given plavix 600
mg PO x 1 prior to transfer to [**Hospital1 18**] for c. cath. He was chest
pain free prior to transfer. Vitals at transfer were BP 145/87
HR 63 SR pOx 100 % on 3 L O2 and RR 20.

He was taken to the c. cath lab showing subtotally occluded LAD
with successful PTCA/stenting with 2.5 x 18 promus stent. LCx
and RCA were patent.

On the floor, patient in NAD without any complaints.

Of note, he was recently hospitalized at [**Name (NI) 75328**] [**Hospital 18806**] Medical
[**Name2 (NI) **] in early [**Name (NI) 547**] for sepsis from a urinary source secondary
to BPH. He completed a course of levofloxacin, was placed on
flomax, and is scheduled to follow-up with urology.

.
On review of systems, he denies any prior history of stroke,
TIA, deep venous thrombosis, pulmonary embolism, bleeding at the
time of surgery, myalgias, joint pains, cough, hemoptysis, black
stools or red stools. He denies exertional buttock or calf pain.
All of the other review of systems were negative.
.
Cardiac review of systems is notable for dyspnea on exertion,
paroxysmal nocturnal dyspnea, orthopnea, ankle edema, syncope or
presyncope. + palpitations two days before the event
.

Past Medical History:
1. CARDIAC RISK FACTORS: Diabetes, +Dyslipidemia, Hypertension
2. CARDIAC HISTORY:
CAD s/p prior PCI
- PERCUTANEOUS CORONARY INTERVENTIONS:
[**Hospital1 2025**] ([**2185**]): Has stent placed to OM2 with ? MI in setting of
shoulder pain. At that time, he was placed on ASA/plavix.
3. OTHER PAST MEDICAL HISTORY:
- BPH with urinary retention
- History of HL
- History of UTI
- Esophageal Dilitation

Social History:
He lives with his wife.
- Tobacco history: none
- ETOH: [**1-6**] glasses of wine/week
- Illicit drugs:  none

Family History:
- Brother died of MI at age 60 (sudden death) while shoveling
snow.
- Mother:  unknown cancer at age [**Age over 90 **]
- Father:  COPD at age 85

Physical Exam:
Tmax: 35.9 ??????C (96.6 ??????F)
Tcurrent: 35.9 ??????C (96.6 ??????F)
HR: 69 (69 - 69) bpm
BP: 125/73(82) {125/73(82) - 125/73(82)} mmHg
RR: 21 (21 - 21) insp/min
SpO2: 98%
Heart rhythm: SR (Sinus Rhythm)

General Appearance: No acute distress
Eyes / Conjunctiva: PERRL
Head, Ears, Nose, Throat: Normocephalic, Poor dentition
Lymphatic: Cervical WNL
Cardiovascular: (S1: Normal), (S2: Normal)
Peripheral Vascular: (Right radial pulse: Present), (Left radial
pulse: Present), (Right DP pulse: Present), (Left DP pulse:
Present)
Respiratory / Chest: (Expansion: Symmetric), (Breath Sounds:
Clear : )
Abdominal: Soft, Non-tender, Bowel sounds present
Extremities: Right lower extremity edema: Absent, Left lower
extremity edema: Absent
Skin:  Warm, No(t) Rash:
Neurologic: Attentive, Follows simple commands, Responds to: Not
assessed, Oriented (to): AAOx2 (not to date fully), Movement:
Not assessed, Tone: Not assessed


Pertinent Results:
I. Cardiology

A. Cath ([**2193-4-26**]) ** PRELIM REPORT **
BRIEF HISTORY:   78 M presented to OSH with chest pain and [**Hospital **]
transferred to [**Hospital1 18**] for emergent cardiac catheterization.

INDICATIONS FOR CATHETERIZATION:
Coronary artery disease, STEMI transfer

PROCEDURE:
Coronary angiography
Conscious Sedation:  was provided with appropriate monitoring
performed by
a member of the nursing staff.

**PTCA RESULTS
 	LAD	 	 	 	 	

PTCA COMMENTS:     Initial angiography reveald a mid LAD 95%
subacute
thrombus. We planned to treat this thrombus with aspiration
thrombectomy/PTCA/stenting and heparin/integrilin given
prophylactically. An XB LAD 4.0 guiding catheter provided good
support
for the procedure and a Prowater wire was advanced into the
distal LAD
with moderate difficulty. We then proceed with an Export AP
aspiration
thrombectomy but unable to deliver device distal to subacute
thrombus.
We then predilated the mid LAD thrombus with an Apex OTW 2.0x8
mm
balloon inflated at 8 atm. We then noted an acute cut-off in the
distal
LAD after flow was re-established and proceeded with cautious
dotting of
the cut-off area with the balloon and distal delivery of NTG via
balloon
with minimal improvement of distal LAD flow. We then stented the
mid LAD
with a Promus Rx 2.5x18 mm drug-eluting stent (DES) post-dilated
with an
NC Quantum Apex MR 2.75x12 mm balloon inflated at 20 atm for 20
sec.
Final angiography revealed normal TIMI 3 flow in the vessel, no
angiographically apparent dissection and 0% residual stenosis in
the
newly deployed stent but acute cut-off in distal LAD showed
diffusely
diseased small apical vesswel that remained unchanged despite
mechanical
dottering and distal NTG delivery via balloon. The R 6Fr femoral
artery
sheath was removed post limited groin angiography and an
Angioseal
closure device was deployed without complications with distal
pulses
confirmed post deployment. The patient left the cath lab
angina-free and
in hemodynamically stable condition.

TECHNICAL FACTORS:
Total time (Lidocaine to test complete) = 59 minutes.
Arterial time = 56 minutes.
Fluoro time = 15.2 minutes.
IRP dose = 733 mGy.
Contrast injected:
          Omnipaque 175 cc total contrast during procedure
Anesthesia:
          1% Lidocaine SC, fentanyl 25 mcg IV, versed 0.5 mg IV
total
Anticoagulation:
          Heparin [**2182**] units, integrilin bolus and infusion

COMMENTS:
1. Emergent coronary angiography revealed a right dominant
systemt. The
LMCA, LCx and RCA were all patent. The LAD revealed a mid 95%
occlusion
with thrombus.
2. Limited resting hemodynamics revealed a SBP of 142 mmHg and a
DBP of
80 mmHg.
3. Successful aspiration thrombectomy/PTCA/stenting of the mid
LAD with
a Promus Rx 2.5x18 mm [**Name Prefix (Prefixes) **] [**Last Name (Prefixes) **]-dilated with an NC 2.75 mm
balloon. Final
angiography revealed normal TIMI 3 flow, no angiographically
apparent
dissection and 0% residual stenosis in the newly deployed stent
with an
abrupt cut-off in the distal LAD unchagned despite mechanical
balloon
dottering and distal NTG delivery via balloon. (see PTCA
comments)
4. R 6Fr femoral artery Angioseal closure device deployed
without
complicatons (see PTCA comments)

FINAL DIAGNOSIS:
1. Severe coronary artery disease with subtotally occluded mid
LAD: see
comments section.
2. Successful aspiration thrombectomy/PTCA/stenting of the mid
LAD with
a Promus Rx 2.5x18 mm [**Name Prefix (Prefixes) **] [**Last Name (Prefixes) **]-dilated with an NC 2.75 mm
balloon. (see
PTCA comments)
3. R 6Fr femoral artery Angioseal closure device deployed
without
complications (see PTCA comments)
4. ASA indefinitely; plavix (clopidogrel) 75 mg daily for at
least 12
months for DES
5. Integrilin gtt for 18 hours post PCI for thrombus and abrupt
cut-off
of distal small vessel apical LAD unchanged despite mechanical
balloon
dottering and distal NTG delivery via balloon

B. TTE ([**2193-4-26**])
Conclusions
The left atrium is elongated. No atrial septal defect is seen by
2D or color Doppler. There is mild symmetric left ventricular
hypertrophy. The left ventricular cavity size is normal. There
is mild to moderate regional left ventricular systolic
dysfunction with basal to mid lateral hypokinesis and distal
septal/distal anterior and apical septal hypokinesis. No masses
or thrombi are seen in the left ventricle. There is no
ventricular septal defect. Right ventricular chamber size and
free wall motion are normal. The diameters of aorta at the
sinus, ascending and arch levels are normal. The aortic valve
leaflets (3) are mildly thickened but aortic stenosis is not
present. Mild (1+) aortic regurgitation is seen. The mitral
valve leaflets are mildly thickened. There is no mitral valve
prolapse. Mild (1+) mitral regurgitation is seen. The tricuspid
valve leaflets are mildly thickened. The pulmonary artery
systolic pressure could not be determined. There is no
pericardial effusion.

C. ECG
No prior ECG available for comparison.
OSH ECG dated [**2193-4-26**] at 9:01 showing ?ectopic atrial rhythm,
NI, leftward axis. STE in V3, V4, and V5.

II. Labs
A. Admission
[**2193-4-26**] 03:15PM BLOOD WBC-7.5 RBC-4.21* Hgb-13.6* Hct-38.9*
MCV-92 MCH-32.3* MCHC-34.9 RDW-12.7 Plt Ct-253
[**2193-4-26**] 03:15PM BLOOD PT-13.4 PTT-27.0 INR(PT)-1.1
[**2193-4-26**] 03:15PM BLOOD Glucose-130* UreaN-15 Creat-1.1 Na-139
K-4.2 Cl-103 HCO3-28 AnGap-12
[**2193-4-26**] 03:15PM BLOOD Calcium-9.4 Phos-3.0 Mg-2.1 Cholest-204*

B. Cardiac
[**2193-4-27**] 05:57AM BLOOD CK(CPK)-426*
[**2193-4-26**] 11:13PM BLOOD CK(CPK)-675*
[**2193-4-27**] 05:57AM BLOOD CK-MB-22* MB Indx-5.2 cTropnT-1.36*
[**2193-4-26**] 11:13PM BLOOD CK-MB-41* MB Indx-6.1*
[**2193-4-26**] 03:15PM BLOOD CK-MB-96* MB Indx-9.2* cTropnT-3.21*

C. Misc
[**2193-4-26**] 03:15PM BLOOD %HbA1c-6.0* eAG-126*
[**2193-4-26**] 03:15PM BLOOD Triglyc-135 HDL-44 CHOL/HD-4.6
LDLcalc-133*

D. Discharge
WBC 4.5 Hgb 11.2 Plt 181 INR 1.2 Na 141 K 4.4 Cl 108 HCO3 29 BUN
20 Cr 1.4 Ca 9.1 Ph 3.2 Mg 2.1

Brief Hospital Course:
79-year-old male with history of CAD and prior PCI with DES to
OM2 at [**Hospital1 2025**] ([**10-7**]) that presented to the ER at OSH with [**Hospital **]
transferred to [**Hospital1 18**], and now s/p successful PTCA/stenting with
DES for LAD lesion.

# STEMI
Patient has known history of CAD given prior stent placement in
OM2. It is uncertain why the patient is not on any cardiac
medications for risk reduction. He presented with chest
discomfort. OSH ECG notable for ectopic atrial rhythm and ST
elevations in V3, V4, and V5 and initial troponin 12.483
(unknown if I or T) and CK-MB 68.5. Cardiac biomarkers indicated
CK-MB 22 and cTrop 1.36. He was transferred to [**Hospital1 18**] for c. cath
with successful PTCA/stenting with DES for 95 % subacute mid-LAD
thrombus.  Final angiography revealed normal TIMI 3 flow and no
angiographically apparent dissection. See cardiac cath report
for full details. Cardiac biomarkers indicated CK-MB 22 and
cTrop 1.36. Post-MI ECHO indicated LVEF 35-40 % withmild to
moderate regional left ventricular systolic dysfunction with
basal to mid lateral hypokinesis and distal septal/distal
anterior and apical septal hypokinesis. This may be suggestive
of another MI given that these wall motion abnormalities do not
necessarily correspond to his LAD lesion.

He was continued on an integrilin infusion for 18 hours post PCI
for thrombus and abrupt cut-off of distal small vessel apical
LAD unchanged despite mechanical balloon dottering and distal
NTG delivery via balloon.

He was placed on aspirin 325 mg PO qD indefinitely, clopidogrel
75 PO qD for at least 12 months for DES. He was started on
crestor given concern for myalgias. He was also started on
metoprolol and lisinopril.

# Hyperlipidemia
Patient was not on lipid-lowering therapy on admission.
Cholesterol panel showing total cholesterol 204, TG 135, HDL 44,
and LDL 133. He was started on statin as above and advised to
initiate lifestyle modifications.

A1c was 6 suggestive of pre-diabetic state.

# RHYTHM: Patient remained in NSR during hospitalization with
telemetry showing bradycardia to low 40s during sleep.

# BPH with urinary retention
Patient was recently hospitalized at [**Name (NI) 75328**] Brothers in the
state of [**Name (NI) 531**] for sepsis from a urinary source in the
setting of urinary retention per provided records from family.
He was continued on flomax during hospitalization and will
follow-up with urology after hospitalization.

CODE: Full

COMM: patient, wife [**Name (NI) **] [**Telephone/Fax (1) 88873**] (H) [**Telephone/Fax (1) 88874**] (C)


Medications on Admission:
- flomax 0.4 mg PO qD
- Multivitamin

Discharge Medications:
1. tamsulosin 0.4 mg Capsule, Ext Release 24 hr Sig: One (1)
Capsule, Ext Release 24 hr PO HS (at bedtime).
2. aspirin 325 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).
3. clopidogrel 75 mg Tablet Sig: One (1) Tablet PO DAILY
(Daily).
Disp:*30 Tablet(s)* Refills:*11*
4. metoprolol succinate 25 mg Tablet Extended Release 24 hr Sig:
One (1) Tablet Extended Release 24 hr PO once a day.
Disp:*30 Tablet Extended Release 24 hr(s)* Refills:*2*
5. lisinopril 5 mg Tablet Sig: 0.5 Tablet PO DAILY (Daily).
Disp:*15 Tablet(s)* Refills:*2*
6. Outpatient Lab Work
Please check Chem-7 and CBC on [**4-1**] at Dr.[**Name (NI) **] office.
7. Crestor 40 mg Tablet Sig: One (1) Tablet PO once a day.
Disp:*30 Tablet(s)* Refills:*2*
8. nitroglycerin 0.4 mg Tablet, Sublingual Sig: One (1) tablet
Sublingual as directed as needed for chest pain.
Disp:*25 tablets* Refills:*0*


Discharge Disposition:
Home

Discharge Diagnosis:
Primary diagnosis:
ST elevation myocardial infarction
Coronary Artery Disease
Acute Kidney Injury
.
Secondary Diagnosis:
Hyperlipidemia


Discharge Condition:
Mental Status: Clear and coherent.
Level of Consciousness: Alert and interactive.
Activity Status: Ambulatory - Independent.


Discharge Instructions:
Dear Mr. [**Known lastname 26762**],

It was a pleasure taking part in your care at [**Hospital1 18**].  You were
transferred here after it was determined that you had suffered a
heart attack prior to arriving at hospital.  You underwent a
cardiac catheterization procedure where a drug eluting stent was
placed in one your heart arteries and you did very well after
this.

You will need to take a number of medications to keep your heart
healthy and make sure the stent stays open.

We have made the following changes to your medications:
START taking aspirin 325 mg and Plavix daily. These medicines
work together to prevent the stent from clotting off. YOu will
need to take these medicines daily for the next year and
possibly longer. Do not stop taking aspirin and Plavix unless
Dr. [**Last Name (STitle) **] says that it is OK.
START taking Rosuvastatin (Crestor) to lower your cholesterol.
YOu will need to have your liver function tested with blood
tests on a regular hasis on this medicine. If you develop muscle
cramps on this medicine, please call Dr. [**Last Name (STitle) **].
START taking Lisinopril to lower your blood pressure and help
your heart recover from the heart attack.
START taking Metoprolol to lower your heart rate and help your
heart recover from the heart attack.
START taking nitroglycerin if you have chest pain at home. Take
one tablet under your tongue, sit down and wait 5 minutes. You
can take another tablet if you still have chest pain but please
call Dr. [**Last Name (STitle) **] if you take any nitroglycerin.
Continue to take Flomax as before.

Followup Instructions:
D'[**Last Name (LF) **],[**First Name3 (LF) **] D. [**Telephone/Fax (1) 22235**]
Appointment already made on [**2193-5-2**] at 11:00 AM
.
Name: [**Last Name (LF) 7526**], [**First Name7 (NamePattern1) **] [**Initial (NamePattern1) **] [**Last Name (NamePattern4) **]
Location: [**First Name5 (NamePattern1) **] [**Last Name (NamePattern1) **] BLDG
Address: 131 ORNAC, [**Apartment Address(1) 88875**], [**Location (un) **],[**Numeric Identifier 17125**]
Phone: [**Telephone/Fax (1) 88876**]
Appt: [**5-16**] at 3:30pm


                             [**Name6 (MD) **] [**Name8 (MD) **] MD, [**MD Number(3) 1905**]
    """
    section_discharge_texts(text)
