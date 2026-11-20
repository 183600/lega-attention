from . import main

from lega.bootstrap import ensure_hash_seed

ensure_hash_seed()

raise SystemExit(main())
