from deepbgc import util

# !!!
# !!!
#
# Don't forget to bump up the DATA_RELEASE_VERSION to deepbgc.__version__ when updating the downloads list
# A data release should be published with a regular code release, with the same release version number
# Code releases can happen more often, so the data release version can lag behind the code version
#
DATA_RELEASE_VERSION = '0.1.0'
#
# !!!
# !!!

PFAM_DB_VERSION = '31.0'
PFAM_DB_FILE_NAME = 'Pfam-A.{}.hmm'.format(PFAM_DB_VERSION)
PFAM_CLANS_FILE_NAME = 'Pfam-A.{}.clans.tsv'.format(PFAM_DB_VERSION)

DOWNLOADS = [
    {
        'url': 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam{}/Pfam-A.hmm.gz'.format(PFAM_DB_VERSION),
        'target': PFAM_DB_FILE_NAME,
        'gzip': True,
        'after': util.run_hmmpress,
        'checksum': '79a3328e4c95b13949a4489b19959fc5',
        'versioned': False
    },
    {
        'url': 'ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam{}/Pfam-A.clans.tsv.gz'.format(PFAM_DB_VERSION),
        'target': PFAM_CLANS_FILE_NAME,
        'gzip': True,
        'checksum': 'a0a4590ffb2b33b83ef2b28f6ead886b',
        'versioned': False
    }
]