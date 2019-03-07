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
        'url': 'https://github.com/Merck/deepbgc/releases/download/v{}/deepbgc.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'deepbgc.pkl',
        'checksum': '7e9218be79ba45bc9adb23bed3845dc1',
        'versioned': True
    },
    {
        'url': 'https://github.com/Merck/deepbgc/releases/download/v{}/clusterfinder_original.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'clusterfinder_original.pkl',
        'dir': 'detector',
        'checksum': '2ca2429bb9bc99a401d1093c376b37aa',
        'versioned': True
    },
    {
        'url': 'https://github.com/Merck/deepbgc/releases/download/v{}/clusterfinder_retrained.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'clusterfinder_retrained.pkl',
        'dir': 'detector',
        'checksum': '65679a3b61c562ff4b84bdb574bb6d93',
        'versioned': True
    },
    {
        'url': 'https://github.com/Merck/deepbgc/releases/download/v{}/clusterfinder_geneborder.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'clusterfinder_geneborder.pkl',
        'dir': 'detector',
        'checksum': 'ca4be7031ae9f70780f17c616a4fa5b5',
        'versioned': True
    },
    {
        'url': 'https://github.com/Merck/deepbgc/releases/download/v{}/product_activity.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'product_activity.pkl',
        'dir': 'classifier',
        'checksum': 'ca4be7031ae9f70780f17c616a4fa5b5',
        'versioned': True
    },
    {
        'url': 'https://github.com/Merck/deepbgc/releases/download/v{}/product_class.pkl'.format(DATA_RELEASE_VERSION),
        'target': 'product_class.pkl',
        'dir': 'classifier',
        'checksum': 'ca4be7031ae9f70780f17c616a4fa5b5',
        'versioned': True
    },
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