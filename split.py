# DISCHARGE SUMMARY
# Section headers with a focus on extracting the 'Brief Hospital Course' section
# Manually reviewed
dis_sum_pre_headers = [
    # sections generally before the hosp course
    'History of present illness:',
    'HPI:',
    'Chief Complaint:'
    'Pertinent Results:',
    'Physical Exam:'
]

dis_sum_brief_hos_course_headers = [
    # hos course section headers...
    'brief hospital course:',
    'hospital course:',
    'concise summary of hospital course by issue/system:',
    'summary of hospital course by systems:',
    'hospital course by systems including pertinent laboratory\ndata:',
    'details of hospital course:',
]

dis_sum_post_headers = [
    # sections after the hosp course section
    '\n\nMedications on Admission:',
    '\n\nCONDITION ON DISCHARGE:',
    '\n\nDictated By:',
    '\n\nDISCHARGE MEDICATIONS:',
    '\n\nDISCHARGE DIAGNOSES:',
    '\n\nCONDITION AT DISCHARGE:',
    '\n\nDischarge Medications:',
    '\n\nDISCHARGE CONDITION:',
    '\n\nDISCHARGE STATUS:',
    '\n\nMEDICATIONS ON DISCHARGE:',
    '\n\nDISCHARGE DIAGNOSIS:',
    '\n\nDISPOSITION:',
    '\n\nDISCHARGE DISPOSITION:',
    '\n\nDischarge Disposition:',
    '\n\nTRANSITIONAL ISSUES:',
    '\n\nTransitional Issues:',
    '\n\nDISCHARGE INSTRUCTIONS:',
    '\n\nACTIVE ISSUES:',
    '\n\nActive Issues:',
    '\n\nPHYSICAL EXAMINATION ON DISCHARGE:',
    '\n\nCODE STATUS:',
    '\n\nTransitional issues:',
    '\n\nCV:',
    '\n\nFINAL DIAGNOSES:',
    '\n\nFEN:',
    '\n\nFOLLOW-UP:',
    '\n\nNOTE:',
    '\n\nCONDITION ON TRANSFER:',
    '\n\nFINAL DIAGNOSIS:',
    '\n\nPHYSICAL EXAMINATION:',
    '\n\nCode:',
    '\n\nLABORATORY DATA:',
    '\n\nActive Diagnoses:',
    '\n\nPROBLEM LIST:',
    '\n\nCODE:',
    '\n\nCHRONIC ISSUES:',
    '\n\nPAST MEDICAL HISTORY:',
    '\n\nID:',
    '\n\nHOSPITAL COURSE:',
    '\n\nPast Medical History:',
    '\n\nICU Course:',
    '\n\nTRANSITIONAL CARE ISSUES:',
    '\n\nActive issues:',
    '\n\nDISCHARGE PHYSICAL EXAMINATION:',
    '\n\nIMPRESSION:',
    '\n\nDISCHARGE PHYSICAL EXAMINATION:  Vital signs:',
    '\n\nFOLLOW UP:',
    '\n\nCONDITION AT TRANSFER:',
    '\n\nMICU course:',
    '\n\nCONDITION OF DISCHARGE:',
    '\n\nICU course:',
    '\n\nMICU Course:',
    '\n\nMEDICATIONS:',
    '\n\nPostoperative course was remarkable for the following:',
    '\n\nPENDING RESULTS:',
    '\n\nACTIVE ISSUES BY PROBLEM:',
    '\n\nMICU COURSE:',
    '\n\nCAUSE OF DEATH:',
    '\n\nACTIVE DIAGNOSES:',
    '\n\nOf note:',
    '\n\nPLAN:',
    '\n\nRECOMMENDATIONS AFTER DISCHARGE:',
    '\n\nCONDITION:',
    '\n\nACUTE ISSUES:',
    '\n\nPlan:',
    '\n\nGI:',
    '\n\nHospital course is reviewed below by problem:',
    '\n\nFINAL DISCHARGE DIAGNOSES:',
    '\n\nDIAGNOSES:',
    '\n\nACTIVE PROBLEMS:',
    '\n\nPROCEDURE:',
    '\n\nMEDICATIONS AT THE TIME OF DISCHARGE:',
    '\n\nDISCHARGE PLAN:',
    '\n\nPENDING LABS:',
    '\n\nDISCHARGE FOLLOWUP:',
    '\n\nChronic Issues:',
    '\n\nHospital course:',
    '\n\nComm:',
    '\n\nFOLLOW-UP INSTRUCTIONS:',
    '\n\nSURGICAL COURSE:',
    '\n\nLABORATORY DATA ON DISCHARGE:',
    '\n\nCode status:',
    '\n\nAddendum:',
    '\n\nACUTE DIAGNOSES:',
    '\n\nLABS ON DISCHARGE:',
    '\n\nTransitions of care:',
    '\n\nFluids, electrolytes and nutrition:',
    '\n\nDISCHARGE INSTRUCTIONS/FOLLOWUP:',
    '\n\nDIAGNOSIS:',
    '\n\nTRANSITIONAL CARE:',
    '\n\nCode Status:',
    '\n\nEvents:',
    '\n\nISSUES:',
    '\n\nFLOOR COURSE:',
    '\n\nFloor Course:',
    '\n\nTransitional:'
]


### Radiology / Echo  Section Headers
# the numbers of occurrences are wrong / different, as this is for MIMIC-III not CXR.
frequent_sections = {
    "preamble": "preamble",  # 227885
    "impression": "impression",  # 187759
    "comparison": "comparison",  # 154647
    "indication": "indication",  # 153730
    "findings": "findings",  # 149842
    "examination": "examination",  # 94094
    "technique": "technique",  # 81402
    "history": "history",  # 45624
    "comparisons": "comparison",  # 8686
    "clinical history": "history",  # 7121
    "reason for examination": "indication",  # 5845
    "notification": "notification",  # 5749
    "reason for exam": "indication",  # 4430
    "clinical information": "history",  # 4024
    "exam": "examination",  # 3907
    "clinical indication": "indication",  # 1945
    "conclusion": "impression",  # 1802
    "conclusions": "impression",
    "concusion": "impression",
    "chest, two views": "findings",  # 1735
    "recommendation(s)": "recommendations",  # 1700
    "type of examination": "examination",  # 1678
    "reference exam": "comparison",  # 347
    "patient history": "history",  # 251
    "addendum": "addendum",  # 183
    "comparison exam": "comparison",  # 163
    "date": "date",  # 108
    "comment": "comment",  # 88
    "findings and impression": "impression",  # 87
    "wet read": "wet read",  # 83
    "comparison film": "comparison",  # 79
    "recommendations": "recommendations",  # 72
    "findings/impression": "impression",  # 47
    "pfi": "history",
    'recommendation': 'recommendations',
    'wetread': 'wet read',
    'summary': 'impression',
    'impresssion': 'impression',
    'impressio': 'impression',
    'ndication': 'impression',  # 1
    'impresson': 'impression',  # 2
    'imprression': 'impression',  # 1
    'imoression': 'impression',  # 1
    'impressoin': 'impression',  # 1
    'imprssion': 'impression',  # 1
    'impresion': 'impression',  # 1
    'imperssion': 'impression',  # 1
    'mpression': 'impression',  # 1
    'impession': 'impression',  # 3
    'findings/ impression': 'impression',  # ,1
    'finding': 'findings',  # ,8
    'findins': 'findings',
    'findindgs': 'findings',  # ,1
    'findgings': 'findings',  # ,1
    'findngs': 'findings',  # ,1
    'findnings': 'findings',  # ,1
    'finidngs': 'findings',  # ,2
    'idication': 'indication',  # ,1
    'reference findings': 'findings',  # ,1
    'comparision': 'comparison',  # ,2
    'comparsion': 'comparison',  # ,1
    'comparrison': 'comparison',  # ,1
    'comparisions': 'comparison'  # ,1
}


impression_section_headers = [k for k,v in frequent_sections.items() if v == 'impression']
