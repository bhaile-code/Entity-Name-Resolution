# Project Structure Overview

This document provides a visual overview of the refactored, modular project structure.

## Directory Tree

```
Entity Name Resolution v2/
│
├── backend/                          # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # Application entry point
│   │   ├── models.py                 # Pydantic data models
│   │   │
│   │   ├── config/                   # Configuration layer
│   │   │   ├── __init__.py
│   │   │   └── settings.py           # Centralized settings
│   │   │
│   │   ├── api/                      # API/HTTP layer
│   │   │   ├── __init__.py
│   │   │   └── routes.py             # Endpoint handlers
│   │   │
│   │   ├── services/                 # Business logic layer
│   │   │   ├── __init__.py
│   │   │   └── name_matcher.py       # Core matching algorithm
│   │   │
│   │   └── utils/                    # Utility layer
│   │       ├── __init__.py
│   │       ├── logger.py             # Logging setup
│   │       └── csv_handler.py        # CSV utilities
│   │
│   ├── tests/                        # Test suite
│   │   ├── __init__.py
│   │   └── test_name_matcher.py
│   │
│   ├── logs/                         # Application logs
│   ├── requirements.txt              # Python dependencies
│   ├── pytest.ini                    # Test configuration
│   └── .env.example                  # Environment template
│
├── frontend/                         # React + Vite frontend
│   ├── src/
│   │   ├── main.jsx                  # Application entry
│   │   ├── App.jsx                   # Main component
│   │   ├── App.css                   # Main styles
│   │   ├── index.css                 # Global styles
│   │   │
│   │   ├── config/                   # Configuration
│   │   │   └── api.config.js         # API settings
│   │   │
│   │   ├── constants/                # Application constants
│   │   │   ├── index.js
│   │   │   └── confidence.js         # Confidence thresholds
│   │   │
│   │   ├── utils/                    # Pure utility functions
│   │   │   ├── index.js
│   │   │   ├── validators.js         # Validation logic
│   │   │   └── formatters.js         # Display formatting
│   │   │
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── index.js
│   │   │   ├── useFileUpload.js      # File upload logic
│   │   │   └── useTableSort.js       # Table sorting logic
│   │   │
│   │   ├── services/                 # External services
│   │   │   ├── api.js                # Backend API client
│   │   │   └── export.js             # File export utilities
│   │   │
│   │   └── components/               # UI components
│   │       ├── FileUpload.jsx        # File upload component
│   │       ├── ResultsTable.jsx      # Results display
│   │       ├── Summary.jsx           # Statistics cards
│   │       └── AuditLog.jsx          # Audit log viewer
│   │
│   ├── public/                       # Static assets
│   ├── index.html                    # HTML template
│   ├── package.json                  # NPM dependencies
│   ├── vite.config.js                # Vite configuration
│   └── .env.example                  # Environment template
│
├── .gitignore                        # Git ignore rules
├── README.md                         # Project documentation
├── CLAUDE.md                         # Development guide for Claude
└── PROJECT_STRUCTURE.md              # This file
```

## Layer Architecture

### Backend Layers (Top to Bottom)

```
┌─────────────────────────────────────────┐
│         HTTP Entry Point                │
│           (main.py)                     │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│          API Layer                      │
│         (api/routes.py)                 │
│  • Route handlers                       │
│  • Request/response validation          │
│  • HTTP error handling                  │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Business Logic Layer              │
│    (services/name_matcher.py)           │
│  • Core matching algorithm              │
│  • No HTTP dependencies                 │
│  • Pure business logic                  │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Utility Layer                   │
│  (utils/csv_handler.py, logger.py)     │
│  • CSV parsing                          │
│  • Logging                              │
│  • Helper functions                     │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Configuration Layer                │
│       (config/settings.py)              │
│  • Environment variables                │
│  • Application settings                 │
│  • Constants                            │
└─────────────────────────────────────────┘
```

### Frontend Layers (Top to Bottom)

```
┌─────────────────────────────────────────┐
│       UI Components Layer               │
│        (components/*.jsx)               │
│  • FileUpload, ResultsTable, etc.       │
│  • Presentation logic                   │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         Hooks Layer                     │
│      (hooks/useFileUpload.js)           │
│  • Reusable stateful logic              │
│  • Component logic extraction           │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Services Layer                    │
│      (services/api.js, export.js)       │
│  • External API communication           │
│  • Side effects                         │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Utilities Layer                   │
│   (utils/validators.js, formatters.js) │
│  • Pure functions                       │
│  • No side effects                      │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   Constants & Configuration             │
│  (constants/*, config/api.config.js)    │
│  • Application constants                │
│  • Configuration values                 │
└─────────────────────────────────────────┘
```

## Data Flow

### Upload and Processing Flow

```
┌─────────────┐
│    User     │
│  (Browser)  │
└──────┬──────┘
       │ 1. Upload CSV
       ▼
┌─────────────────────┐
│  FileUpload.jsx     │
│  (Frontend)         │
│  • Validates file   │
│  • Shows progress   │
└──────┬──────────────┘
       │ 2. POST /api/process
       ▼
┌─────────────────────┐
│   api/routes.py     │
│   (Backend API)     │
│  • Receives file    │
│  • Validates format │
└──────┬──────────────┘
       │ 3. Parse CSV
       ▼
┌─────────────────────┐
│  CSVHandler         │
│  (Backend Utils)    │
│  • Extracts names   │
│  • Validates data   │
└──────┬──────────────┘
       │ 4. Process names
       ▼
┌─────────────────────┐
│  NameMatcher        │
│  (Backend Service)  │
│  • Groups similar   │
│  • Selects canonical│
│  • Generates audit  │
└──────┬──────────────┘
       │ 5. Return results
       ▼
┌─────────────────────┐
│  ResultsTable.jsx   │
│  (Frontend)         │
│  • Displays results │
│  • Allows export    │
└─────────────────────┘
```

## Module Dependencies

### Backend Import Graph

```
main.py
  ├── config.settings
  ├── api.router
  │   ├── models
  │   ├── services.NameMatcher
  │   │   └── config.settings
  │   ├── utils.CSVHandler
  │   └── utils.logger
  │       └── config.settings
  └── utils.logger
```

### Frontend Import Graph

```
App.jsx
  ├── components/*
  │   ├── hooks/*
  │   │   ├── services/api.js
  │   │   │   └── config/api.config.js
  │   │   └── utils/*
  │   │       └── constants/*
  │   ├── services/*
  │   └── utils/*
  └── constants/*
```

## Key Design Principles

### Separation of Concerns

- **Backend**: Config → Utils → Services → API → Main
- **Frontend**: Constants → Utils → Services → Hooks → Components → App

### Dependency Direction

- Dependencies flow **inward** (from outer layers to inner layers)
- Inner layers (utils, config) have no dependencies on outer layers
- Makes testing easier (can test services without HTTP)

### Modularity Benefits

1. **Easy to test**: Each layer can be tested independently
2. **Easy to change**: Changes localized to specific layers
3. **Easy to understand**: Clear responsibility for each module
4. **Easy to extend**: Add new features without touching existing code
5. **Easy to iterate**: Prototype flexibility with good structure

### Import Rules

**Backend**:
- ✅ `services` can import from `utils`, `config`
- ✅ `api` can import from `services`, `utils`, `config`, `models`
- ❌ `utils` should NOT import from `services` or `api`
- ❌ `config` should NOT import from any other app module

**Frontend**:
- ✅ `components` can import from `hooks`, `services`, `utils`, `constants`, `config`
- ✅ `hooks` can import from `services`, `utils`, `constants`
- ✅ `services` can import from `utils`, `constants`, `config`
- ❌ `utils` should NOT import from `services`, `hooks`, `components`
- ❌ `constants` should NOT import from any other module

## Quick Reference

### Adding New Features

| What | Where | Dependencies |
|------|-------|--------------|
| New algorithm | `backend/app/services/` | config, utils |
| New API endpoint | `backend/app/api/routes.py` | services, utils, models |
| New config value | `backend/app/config/settings.py` | none |
| New utility | `backend/app/utils/` | config only |
| New UI component | `frontend/src/components/` | hooks, utils, services |
| New custom hook | `frontend/src/hooks/` | services, utils |
| New constant | `frontend/src/constants/` | none |
| New API call | `frontend/src/services/api.js` | config |

### File Naming Conventions

- **Backend**: `snake_case.py` (Python convention)
- **Frontend**: `camelCase.js` for files, `PascalCase.jsx` for components
- **Constants**: `UPPER_SNAKE_CASE` for values
- **Functions**: `camelCase` (JS) or `snake_case` (Python)
- **Classes**: `PascalCase` (both)

---

**Note**: This structure balances prototype simplicity with professional code organization. It's designed to be easy to iterate on while maintaining good engineering practices.
