# CORE REQUIREMENTS (Source of Truth)

> **Purpose**: This document defines mandatory patterns and standards for
> PosalPro MVP2 development. Follow Required items strictly; Recommended items
> can be adapted with documented reasoning.

## Table of Contents

1. [Required Safeguards](#required-safeguards) ⚠️ **CRITICAL**
2. [Quick Start Guide](#quick-start-guide)
3. [Architecture Overview](#architecture-overview)
4. [Mandatory Patterns](#mandatory-patterns)
5. [API & Routes](#api--routes)
6. [Performance & Optimization](#performance--optimization)
7. [Quality Standards](#quality-standards)
   - [Code Simplicity & Maintainability](#1-code-simplicity--maintainability-mandatory)
   - [TypeScript Compliance](#2-typescript-compliance-required)
   - [Testing Requirements](#3-testing-requirements)
   - [Accessibility](#4-accessibility-wcag-21-aa)
   - [Security Requirements](#5-security-requirements)
   - [Pre-Implementation Checklist](#6-pre-implementation-checklist)
   - [Component Design Standards](#7-component-design-standards-new-components)
8. [Platform Configuration](#platform-configuration)
9. [Reference Implementation](#reference-implementation)
10. [Anti-Patterns & Pitfalls](#anti-patterns--pitfalls)
11. [Configuration Reference](#configuration-reference)
12. [Related Documents](#related-documents)
13. [How to Use This Document](#how-to-use-this-document)

---

## Required Safeguards

> ⚠️ **CRITICAL**: These items are non-negotiable and must be preserved in all
> future changes.

### Auth/JWT Secret Consistency (MANDATORY)

All signing and verification must resolve the secret via a single source of
truth:

```typescript
// ✅ CORRECT: Use getAuthSecret() everywhere
import { getAuthSecret } from '@/lib/auth/secret';

const secret = getAuthSecret(); // NextAuth, JWT, getToken all use this
```

**References**: `src/lib/auth/secret.ts`, `src/lib/auth.ts`,
`src/middleware.ts`, `src/app/api/auth/debug/route.ts`

### Auth Observability (REQUIRED)

When `AUTH_DEBUG` or `NEXTAUTH_DEBUG` is true:

- NextAuth debug logging must be enabled
- Structured logging routed through project logger
- Request-ID correlation for traceability

**References**: `src/lib/auth.ts`, `src/lib/logger.ts`

### Session Integrity (REQUIRED)

With valid token (Authorization header or session cookie):

- `/api/auth/debug` must return populated token
- `/api/auth/session` must return populated session

**References**: `src/app/api/auth/debug/route.ts`,
`src/app/api/auth/[...nextauth]/route.ts`

### Environment Validation (MANDATORY)

NextAuth-related variables must be validated and documented:

- `NEXTAUTH_SECRET` (32+ characters required)
- `NEXTAUTH_URL` (full URL with protocol)
- `JWT_SECRET` (optional, for parity)

Validate via `@t3-oss/env-nextjs` or Zod in `src/env.mjs` and `src/env.ts`.

### Cookie Security (PRODUCTION REQUIRED)

```typescript
// ✅ CORRECT: Secure cookie settings
{
  useSecureCookies: process.env.NODE_ENV === 'production',
  cookies: {
    sessionToken: {
      name: process.env.NODE_ENV === 'production'
        ? '__Secure-next-auth.session-token'
        : 'next-auth.session-token',
      options: {
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        secure: process.env.NODE_ENV === 'production',
      },
    },
  },
}
```

**Reference**: `src/lib/auth.ts`

### PDF Preview Stability (REQUIRED)

```typescript
// ✅ CORRECT: Single react-pdf instance, same-origin worker
import { pdfjs } from 'react-pdf';

pdfjs.GlobalWorkerOptions.workerPort = workerPort; // Bundled worker

// Guard rendering
{numPages > 0 && <Page pageNumber={1} />}

// Proxy network PDFs through API
<Document file="/api/documents?url=https://..." />
```

**References**: `src/components/providers/QueryProvider.tsx`,
`src/components/products/DocumentPreview.tsx`, `src/app/api/documents/route.ts`

### Prisma Engine/URL Alignment (MANDATORY)

```typescript
// ✅ CORRECT: Local PostgreSQL (standard client)
DATABASE_URL = 'postgresql://user:pass@localhost:5432/db';
// No PRISMA_CLIENT_ENGINE_TYPE needed

// ❌ WRONG: Protocol mismatch causes errors
DATABASE_URL = 'postgresql://...'; // Local URL
PRISMA_CLIENT_ENGINE_TYPE = 'dataproxy'; // Data Proxy mode (mismatch!)
```

**Validation**: `src/lib/db/prisma.ts` includes diagnostics and error messages.

---

## Quick Start Guide

### Essential Commands

```bash
# Development
npm run dev:smart              # Start dev server (kills port 3000 first)
npm run type-check            # TypeScript validation (0 errors required)
npm run audit:duplicates      # Check for duplicate implementations

# Database
npx prisma generate          # Generate Prisma client
npx prisma db push          # Push schema changes

# Testing
npm test                    # Run test suite
npx tsx scripts/app-cli.ts # CLI for testing features
```

### Critical Patterns (5-Minute Essentials)

**1. Feature Module Structure**

```
src/features/[domain]/
├── schemas.ts        # All Zod schemas
├── keys.ts          # React Query keys
├── hooks/           # React Query hooks
└── index.ts         # Consolidated exports
```

**2. Service Layer (BaseService)**

```typescript
// All services extend BaseService
export class ProductService extends BaseService {
  constructor() {
    super('ProductService', '/api/products');
  }

  async getProducts(params: ProductQuery): Promise<ProductList> {
    return this.executeWithLogging(
      'getProducts',
      () => this.get<ProductList>(`?${this.buildQueryString(params)}`),
      { params, userStory: 'US-4.1', hypothesis: 'H5' }
    );
  }
}
```

### 10. Offline Capability & Service Worker Governance

- Service worker registration MUST flow through `ServiceWorkerManager` to
  enforce config-driven updates, background sync, and cache telemetry
  (`src/lib/offline/ServiceWorkerManager.ts`).
- UI wiring uses `ServiceWorkerProvider` so offline toasts and update prompts
  remain feature-flagged via `NEXT_PUBLIC_ENABLE_SW`
  (`src/components/providers/ServiceWorkerProvider.tsx`).

**3. React Query Hook**

```typescript
export function useProducts(params: ProductQuery) {
  return useQuery({
    queryKey: productKeys.list(params),
    queryFn: () => productService.getProducts(params),
    staleTime: 30000, // 30s
    gcTime: 120000, // 2min
  });
}
```

**4. API Route**

```typescript
export const GET = createRoute(
  { query: ProductQuerySchema, permissions: ['products:read'] },
  async ({ query, user, requestId }) => {
    const products = await productService.listProductsCursor(query);
    return ok(products);
  }
);
```

**5. Route Boundaries** (REQUIRED for every route)

```
src/app/(dashboard)/[feature]/
├── page.tsx           # Main route
├── loading.tsx        # Loading UI (required)
└── error.tsx          # Error boundary (required)
```

### Critical Don'ts

❌ **DON'T**: Write clever code - optimize for readability (2-year dev test) ❌
**DON'T**: Create new APIs when existing ones work ❌ **DON'T**: Store server
state in Zustand (use React Query) ❌ **DON'T**: Use `console.log` (use
structured logger) ❌ **DON'T**: Manual response envelope handling ❌ **DON'T**:
Skip route boundaries (loading/error files)

---

## Architecture Overview

### Data Flow (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│ UI Components (React)                                       │
│ - Form handling, user interactions, accessibility          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ State Management Layer                                      │
│                                                             │
│  Zustand (UI State)          React Query (Server State)    │
│  - Filters, selections       - Data fetching & caching     │
│  - Modal state              - Mutations & invalidation     │
│  src/lib/store/*            src/features/*/hooks/*         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Service Layer (Two-Tier Architecture)                      │
│                                                             │
│  Frontend Services          Database Services              │
│  src/services/*            src/lib/services/*              │
│  - HTTP client             - Prisma ORM                    │
│  - Extends BaseService     - Business logic                │
│  - React Query integration - Transactions                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ API Routes (Thin Boundaries)                               │
│ src/app/api/[resource]/route.ts                           │
│ - createRoute wrapper (auth, validation, idempotency)     │
│ - Permission checks (validateApiPermission)                │
│ - Delegates to database services                          │
│ ❌ NO direct Prisma imports, NO business logic            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Database Layer (PostgreSQL + Prisma)                       │
│ - ACID transactions, indexes, connection pooling           │
│ - Schema: prisma/schema.prisma (source of truth)          │
└─────────────────────────────────────────────────────────────┘

Cross-Cutting: ErrorHandlingService, Structured Logging, Redis Cache
```

### Provider Hierarchy

```
RootLayout
└── QueryProvider (React Query)
    └── AuthProvider (NextAuth session)
        └── ErrorProvider (Global error handling)
            └── ProtectedLayout (Auth gating)
                └── AppLayout (Header, Sidebar, Navigation)
                    └── Page Content
```

---

## Mandatory Patterns

### 1. Feature-Based Organization

**Required Structure:**

```
src/features/[domain]/
├── schemas.ts        # All Zod schemas, types, validation
├── keys.ts          # Centralized React Query keys
├── hooks/           # React Query hooks
│   └── use[Domain].ts
└── index.ts         # Consolidated exports
```

**Example: Products Feature**

```typescript
// src/features/products/schemas.ts
export const ProductSchema = z.object({
  id: z.string(),
  name: z.string().min(1),
  price: z.number().min(0),
  // ... align with Prisma schema
});

export const ProductQuerySchema = z.object({
  search: z.string().optional(),
  category: z.string().optional(),
  limit: z.coerce.number().min(1).max(100).default(20),
  cursor: z.string().optional(),
});

// src/features/products/keys.ts
export const qk = {
  products: {
    all: ['products'] as const,
    lists: () => [...qk.products.all, 'list'] as const,
    list: params => [...qk.products.lists(), params] as const,
    byId: (id: string) => [...qk.products.all, 'byId', id] as const,
  },
} as const;

// src/features/products/index.ts
export { qk as productKeys } from './keys';
export * from './schemas';
export * from './hooks';
```

### 2. Service Layer Architecture

#### Two-Tier Pattern (REQUIRED)

**Frontend Services** (`src/services/`)

- HTTP client services for React Query integration
- Extend `BaseService` for consistency
- Stateless, API communication only

**Database Services** (`src/lib/services/`)

- Direct Prisma access for API routes
- Complex queries, transactions, business logic
- Server-side only

#### BaseService Pattern (REQUIRED for Frontend Services)

```typescript
// src/services/productService.ts
import { BaseService } from '@/lib/services/BaseService';
import { ProductQuerySchema, ProductCreateSchema } from '@/features/products';

export class ProductService extends BaseService {
  constructor() {
    super('ProductService', '/api/products');
  }

  async getProducts(params: ProductQuery): Promise<ProductList> {
    const validated = ProductQuerySchema.parse(params);
    return this.executeWithLogging(
      'getProducts',
      () => this.get<ProductList>(`?${this.buildQueryString(validated)}`),
      { params: validated, userStory: 'US-4.1', hypothesis: 'H5' }
    );
  }

  async createProduct(data: ProductCreate): Promise<Product> {
    const validated = ProductCreateSchema.parse(data);
    return this.executeWithLogging(
      'createProduct',
      () => this.post<Product>('/', validated),
      { userStory: 'US-4.1' }
    );
  }
}

export const productService = new ProductService();
```

**BaseService Benefits:**

- Automatic error handling via `ErrorHandlingService`
- Structured logging with request correlation
- Performance tracking
- Consistent patterns across all services

**Database Service Pattern:**

```typescript
// src/lib/services/productService.ts
export class ProductService {
  async listProductsCursor(filters: ProductFilters) {
    const where = this.buildWhereClause(filters);
    return prisma.product.findMany({ where, take: filters.limit });
  }

  async createProductWithValidation(data: CreateProductData): Promise<Product> {
    // Business logic, validation, transactions
    return prisma.product.create({ data });
  }
}
```

### 3. State Management

**Zustand: UI State Only** (src/lib/store/)

```typescript
interface ProductUIState {
  filters: { search: string; category?: string };
  selectedIds: string[];
  setFilters: (filters) => void;
  toggleSelection: (id: string) => void;
}

export const useProductUI = create<ProductUIState>(set => ({
  filters: { search: '' },
  selectedIds: [],
  setFilters: filters => set({ filters }),
  toggleSelection: id =>
    set(state => ({
      selectedIds: state.selectedIds.includes(id)
        ? state.selectedIds.filter(x => x !== id)
        : [...state.selectedIds, id],
    })),
}));

// Usage: Individual selectors with useShallow
export const useFilters = () => useProductUI(state => state.filters);
export const useSelectedIds = () => useProductUI(state => state.selectedIds);
```

**React Query: Server State** (src/features/\*/hooks/)

```typescript
export function useProducts(params: ProductQuery) {
  return useQuery({
    queryKey: productKeys.list(params),
    queryFn: () => productService.getProducts(params),
    staleTime: 30000, // 30s - data considered fresh
    gcTime: 120000, // 2min - cache retention
    refetchOnWindowFocus: false,
    refetchOnReconnect: true,
    retry: 2,
    retryDelay: attempt => Math.min(1000 * 2 ** attempt, 5000),
  });
}
```

### 4. Error Handling

**Centralized ErrorHandlingService** (REQUIRED)

```typescript
import { ErrorHandlingService, StandardError, ErrorCodes } from '@/lib/errors';
import { logInfo, logError } from '@/lib/logger';

try {
  const result = await operation();
  logInfo('Operation successful', {
    component: 'ServiceName',
    operation: 'methodName',
    requestId, // Auto-correlated
  });
} catch (error) {
  const errorService = ErrorHandlingService.getInstance();
  const processed = errorService.processError(
    error,
    'Operation failed',
    ErrorCodes.SYSTEM.UNKNOWN,
    { component: 'ServiceName', operation: 'methodName', requestId }
  );

  logError('Operation failed', processed, {
    component: 'ServiceName',
    errorCode: processed.code,
  });

  throw processed;
}
```

**Error Categories:**

- `SYSTEM`: Internal errors, unknown errors
- `DATA`: Database errors, not found, validation
- `API`: Request failed, timeout, network
- `VALIDATION`: Invalid input, schema errors
- `AUTH`: Unauthorized, forbidden
- `BUSINESS`: Business rule violations

### 5. Schema & Validation

**Single Source of Truth** (REQUIRED)

- All Zod schemas in `src/features/[domain]/schemas.ts`
- Database schema in `prisma/schema.prisma` is the source of truth
- Field names must match Prisma schema exactly

**Query Parameter Alignment:**

```typescript
// ✅ CORRECT: Direct parameters for validated fields
const params = { category: 'Electronics', limit: 50 };
// Results in: /api/products?category=Electronics&limit=50

// ❌ WRONG: Wrapping in filters object
const params = { filters: { category: 'Electronics' }, limit: 50 };
// Results in: /api/products?filters[category]=Electronics (breaks validation)
```

**Nullable Pattern Consistency:**

```typescript
// ✅ CORRECT: Consistent nullable patterns
export const CustomerSchema = z.object({
  phone: z.string().nullable().optional(),
});

export const CustomerUpdateSchema = z.object({
  phone: z.string().nullable().optional(), // Matches main schema
});
```

---

## API & Routes

### 1. createRoute Wrapper (REQUIRED)

```typescript
import { createRoute } from '@/lib/api/route';
import { ok } from '@/lib/api/response';

export const GET = createRoute(
  {
    requireAuth: true,
    permissions: ['products:read'],
    query: ProductQuerySchema,
  },
  async ({ query, user, requestId }) => {
    const products = await productService.listProductsCursor(query);
    return ok(products);
  }
);

export const POST = createRoute(
  {
    roles: ['admin', 'manager'],
    permissions: ['products:create'],
    body: ProductCreateSchema,
    idempotency: true, // Auto-caches by Idempotency-Key header
  },
  async ({ body, user, requestId }) => {
    const product = await productService.createProductWithValidation(body);
    return ok(product, 201);
  }
);
```

**createRoute Features:**

- Automatic auth validation
- Zod schema validation
- Permission checking
- Request-ID propagation
- Idempotency protection
- Standardized error responses

### 2. Route Boundaries (REQUIRED)

**Every route must have:**

```
src/app/(dashboard)/products/
├── page.tsx           # Main component
├── loading.tsx        # Loading UI (required)
└── error.tsx          # Error boundary (required)
```

**Example loading.tsx:**

```typescript
export default function Loading() {
  return <ProductListSkeleton />;
}
```

**Example error.tsx:**

```typescript
'use client';
export default function Error({ error, reset }: ErrorPageProps) {
  return <ErrorDisplay error={error} onRetry={reset} />;
}
```

### 3. RBAC & Permissions

**Dynamic Permissions** (REQUIRED for new routes)

```typescript
// ✅ CORRECT: Dynamic database checking
export const GET = createRoute({
  permissions: ['proposals:read'], // Checked from database
}, async ({ user }) => {
  // Handler
});

// ❌ AVOID: Hard-coded entitlements (only for tenant licensing)
export const GET = createRoute({
  entitlements: ['feature.proposals.read'], // Only for admin/tenant features
}, ...);
```

**Permission Format:** `<resource>:<action>`

- Examples: `proposals:read`, `customers:create`, `products:update`
- Scopes: ALL, TENANT, DEPARTMENT, OWN
- Administrator bypass: Automatic for System Administrator role

### 4. Response Envelopes

**API Routes Return:**

```typescript
import { ok, error } from '@/lib/api/response';

// Success
return ok({ items, total, nextCursor });

// Error (handled by createRoute automatically)
throw new StandardError({
  message: 'Product not found',
  code: ErrorCodes.DATA.NOT_FOUND,
});
```

**Frontend Services Unwrap:**

```typescript
// ✅ HTTP client returns unwrapped data
const product = await http.get<Product>(`/api/products/${id}`);
return product; // Already unwrapped

// ❌ Don't manually unwrap
const response = await http.get<ApiResponse<Product>>(...);
if (response.ok) return response.data; // HTTP client does this
```

---

## Performance & Optimization

> **Philosophy**: Optimize for user experience (speed) and operational
> efficiency (cost). Every pattern below addresses specific performance goals.

### 1. Less I/O (Minimize Network Requests)

**Parallel Data Loading** (REQUIRED)

```typescript
// ✅ CORRECT: Load all data in parallel (500ms total)
function useUnifiedProductData() {
  const [products, stats, categories] = React.useMemo(
    () => [
      useInfiniteProducts(params),
      useProductStats(),
      useProductCategories(),
    ],
    [params]
  );

  return { products, stats, categories };
}

// ❌ WRONG: Sequential loading (waterfall effect: 4169ms)
const products = await fetchProducts();
const stats = await fetchStats();
const categories = await fetchCategories();
```

**Single Query Aggregation** (70% faster)

```sql
-- ✅ CORRECT: 1 database query instead of 6
SELECT json_build_object(
  'totalUsers', (SELECT COUNT(*) FROM users),
  'activeUsers', (SELECT COUNT(*) FROM users WHERE status = 'ACTIVE'),
  'totalProposals', (SELECT COUNT(*) FROM proposals)
) as aggregated_data

-- ❌ WRONG: 6 separate queries (6x slower)
```

**Benefits**: 4x faster page loads, reduced backend load, better user experience

**Smart Payload Selection** (REQUIRED)

- Endpoints that support scoped payloads (currently products, proposals) must
  accept `scope=lite` and return lean selects (no heavy relations) when
  requested.
- React Query keys MUST include the scope to isolate caches
  (`productKeys.products.byId(id, 'lite')`).
- Inline editors, sidebars, and wizard warmups MUST request `scope: 'lite'` with
  `includeRelations: false` unless specific minimal fields are required.
- Default to `scope: 'full'` only for detail views or workflows that truly
  require full relations.

### 2. Less Frontend Load (Save User Resources)

**Lazy Loading & Code Splitting** (REQUIRED)

```typescript
// ✅ CORRECT: Dynamic imports (~500KB bundle reduction)
const ProductChart = dynamic(() => import('@/components/products/ProductChart'), {
  ssr: false,
  loading: () => <ChartSkeleton />,
});

// ✅ CORRECT: Lazy load heavy libraries on demand
const loadChart = async () => {
  const { Chart } = await import('chart.js');
  return Chart;
};
```

**Memoization** (Reduce Re-renders)

```typescript
// ✅ CORRECT: Memoize expensive computations
const productMetrics = useMemo(() => {
  return products.map(p => calculateMetrics(p));
}, [products]); // Only recalculate when products change

// ✅ CORRECT: Memoize components
const ProductCard = memo(({ product }) => {
  return <Card>{product.name}</Card>;
});
```

**Benefits**: Faster page loads, lower CPU usage, better mobile performance,
reduced memory consumption

### 3. Less Backend Load (Fast & Responsive App)

**Optimistic Updates** (75% API call reduction)

```typescript
// ✅ CORRECT: Update UI instantly, single invalidation
export function useUpdateProduct() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: data => productService.updateProduct(data.id, data),
    onSuccess: (updatedProduct, variables) => {
      // 1. Instant UI update (use mutation response)
      queryClient.setQueryData(productKeys.byId(variables.id), updatedProduct);

      // 2. Single targeted invalidation
      queryClient.invalidateQueries({
        queryKey: productKeys.lists(),
        exact: true,
      });
    },
  });
}

// ❌ WRONG: Extra API call after mutation
const fresh = await service.get(id); // Unnecessary extra call
queryClient.setQueryData(keys.byId(id), fresh);
```

**React Query Configuration** (Smart Caching)

```typescript
{
  staleTime: 30000,        // 30s - prevent unnecessary refetches
  gcTime: 120000,          // 2min - cache retention
  refetchOnWindowFocus: false, // Don't refetch on tab switch
  refetchOnReconnect: true,    // Refetch on connection restore
  enabled: true,            // Conditional fetching (route-aware)
  refetchInterval: false,   // Smart polling (only when needed)
}
```

**Route-Aware Data Fetching** (REQUIRED)

```typescript
// ✅ CORRECT: Conditional loading based on route
const pathname = usePathname();
const shouldLoad =
  pathname.startsWith('/dashboard') || pathname.startsWith('/proposals');
const data = useQuery({
  queryKey: ['data'],
  queryFn: fetchData,
  enabled: shouldLoad, // Only load when relevant
});
```

**Shared Context Pattern** (REQUIRED for duplicate elimination)

```typescript
// ✅ CORRECT: Provider pattern for shared queries
export function UserPreferencesProvider({ children }) {
  const preferences = useUserPreferences(); // Called ONCE
  return <UserPreferencesContext.Provider value={preferences}>{children}</UserPreferencesContext.Provider>;
}

// Components use context instead of direct hook
const { data: preferences } = useUserPreferencesContext();
```

**Benefits**: 75% fewer API calls, instant UI feedback, reduced server load, 30%
performance improvement

### 4. Less Compute (Minimize Cloud Costs)

**Request Deduplication** (Thundering Herd Prevention)

```typescript
// ✅ CORRECT: Prevent duplicate concurrent requests (70-90% reduction during spikes)
export class ProductService {
  private pendingRequests = new Map<string, Promise<ProductList>>();

  async listProducts(filters: ProductFilters): Promise<ProductList> {
    const cacheKey = JSON.stringify(filters);

    // Return existing promise if request in flight
    const pending = this.pendingRequests.get(cacheKey);
    if (pending) return pending;

    // Execute and cache promise
    const promise = this.executeQuery(filters).finally(() =>
      this.pendingRequests.delete(cacheKey)
    );

    this.pendingRequests.set(cacheKey, promise);
    return promise;
  }
}
```

**Field Selection** (30-60% payload reduction)

```typescript
// ✅ CORRECT: Request only needed fields
const params = { fields: 'id,name,price' }; // Minimal data
// API returns: { id, name, price } only

// ❌ WRONG: Fetch all fields when only need few
const product = await get('/api/products/123'); // Returns everything
```

**Database Query Optimization**

```typescript
// ✅ CORRECT: Slow query monitoring
const start = performance.now();
const results = await prisma.product.findMany();
const duration = performance.now() - start;

if (duration > 750) {
  logWarn('Slow query detected', {
    durationMs: duration,
    threshold: 750,
  });
}
```

**Benefits**: Lower database costs, reduced bandwidth, cheaper infrastructure,
faster queries

### 5. No Duplicate Requests (Prevent Unnecessary Calls)

**Cursor Pagination** (REQUIRED)

```typescript
// ✅ CORRECT: Cursor-based pagination (efficient)
export function useInfiniteProducts(params) {
  return useInfiniteQuery({
    queryKey: productKeys.list(params),
    queryFn: ({ pageParam }) =>
      productService.getProducts({
        ...params,
        cursor: pageParam,
      }),
    initialPageParam: null,
    getNextPageParam: lastPage => lastPage.nextCursor || undefined,
  });
}

// ❌ WRONG: Offset pagination (deprecated, slow on large datasets)
```

**Idempotency Protection**

```typescript
// ✅ CORRECT: Automatic idempotency via createRoute
export const POST = createRoute(
  {
    body: ProductCreateSchema,
    idempotency: true, // Prevents duplicate creates
  },
  async ({ body }) => {
    // If Idempotency-Key header present, auto-cached
    return ok(await productService.create(body));
  }
);
```

**Benefits**: Prevent accidental duplicate operations, consistent results,
reduced costs

### 6. Backend-Frontend Data Consistency

**Single Source of Truth** (REQUIRED)

```typescript
// ✅ CORRECT: React Query provides consistent cached data
const { data: product } = useProduct(productId);
// All components reading this productId get same cached data

// Header component
const { data: product } = useProduct(productId); // From cache

// Detail component
const { data: product } = useProduct(productId); // Same cached data

// Sidebar component
const { data: product } = useProduct(productId); // Same cached data
```

**Automatic Sync After Mutations**

```typescript
// ✅ CORRECT: Update all instances automatically
onSuccess: updated => {
  // This updates cache - all components using this ID see update
  queryClient.setQueryData(productKeys.byId(id), updated);
};
```

**Benefits**: No stale data across components, single source of truth,
predictable UI state

### 7. Data Persistence Optimization (Fast Load Feel)

**Zustand Persistence** (Form Data Across Steps)

```typescript
// ✅ CORRECT: Persist wizard data across steps
export const useProposalStore = create<ProposalState>()(
  persist(
    set => ({
      wizardData: {},
      setStepData: (step, data) =>
        set(state => ({
          wizardData: { ...state.wizardData, [step]: data },
        })),
    }),
    { name: 'proposal-wizard' } // localStorage persistence
  )
);
```

**React Query Placeholder Data**

```typescript
// ✅ CORRECT: Keep previous data while loading new data
{
  placeholderData: (previousData) => previousData,
  // User sees old data during refetch (no empty state flicker)
}
```

**Cache Warming** (Instant Loads)

```typescript
// ✅ CORRECT: Pre-warm cache after mutations
await cachedExec(
  `products:list:${tenantId}:first-page`,
  () => productService.list({ tenantId, limit: 20 }),
  300 // 5min TTL
);
// Next user gets instant response from cache
```

**Benefits**: No empty states, instant perceived load times, better UX

### 8. Data Integrity (Consistent Form Data)

**Wizard State Persistence** (REQUIRED for Multi-Step Forms)

```typescript
// ✅ CORRECT: Data persists across wizard steps
interface ProposalWizardState {
  step1Data: BasicInfo;
  step2Data: ProductSelection;
  step3Data: TeamAssignment;

  setStepData: (step: number, data: unknown) => void;
  getStepData: (step: number) => unknown;
}

// Step 2 can access Step 1 data
const step1Data = useProposalStore(state => state.step1Data);
```

**Form State Validation**

```typescript
// ✅ CORRECT: Validate complete data before submission
const validateWizard = () => {
  const allSteps = [step1Data, step2Data, step3Data];
  return allSteps.every(step => step && Object.keys(step).length > 0);
};
```

**Benefits**: No data loss between steps, consistent form state, better user
experience

### 9. Caching Strategy

**Redis Cache** (src/lib/redis.ts)

```typescript
// ✅ CORRECT: Multi-layer caching
import { getCache, setCache } from '@/lib/redis';

// Layer 1: React Query (client-side, 30s)
// Layer 2: Redis (server-side, 5min)
const cached = await getCache<ProductList>('products:list');
if (cached) return cached;

const fresh = await fetchProducts();
await setCache('products:list', fresh, 300);
return fresh;
```

**Cache TTLs by Data Type:**

| Data Type       | TTL        | Rationale            |
| --------------- | ---------- | -------------------- |
| Session & Auth  | 180-300s   | Security-focused     |
| User Data       | 900-1200s  | Moderate changes     |
| API Responses   | 300s       | Balanced performance |
| Product Catalog | 1800s      | Relatively static    |
| Cache Warming   | 3600-7200s | Prefetched data      |

**Cache Invalidation**

```typescript
// ✅ CORRECT: Targeted invalidation
queryClient.invalidateQueries({ queryKey: productKeys.lists(), exact: true });

// ❌ WRONG: Broad invalidation (invalidates unrelated queries)
queryClient.invalidateQueries({ queryKey: ['products'] });
```

**Cache-First Service Pattern** (REQUIRED for high-frequency services)

```typescript
// ✅ CORRECT: Enforce caching via public API
export class PermissionService {
  // Public method - always cached
  static async getUserPermissions(userId: string): Promise<string[]> {
    return this.getUserPermissionsCached(userId);
  }

  // Private - DB access only (prevents cache bypass)
  private static async fetchUserPermissionsFromDb(
    userId: string
  ): Promise<string[]> {
    return prisma.userRole.findMany({
      /* ... */
    });
  }

  // Cache layer with Redis/memory fallback
  static async getUserPermissionsCached(userId: string): Promise<string[]> {
    const cached = await getCache(cacheKey);
    if (cached) return cached;

    const fresh = await this.fetchUserPermissionsFromDb(userId);
    await setCache(cacheKey, fresh, TTL);
    return fresh;
  }
}

// ❌ WRONG: Cache is optional, can be bypassed
export class PermissionService {
  static async getUserPermissions(userId: string): Promise<string[]> {
    return prisma.userRole.findMany({
      /* ... */
    }); // Direct DB, no cache
  }
}
```

**Apply to**: Permissions, roles, auth checks, feature flags (>100:1 read/write
ratio) **Reference**: `src/lib/services/PermissionService.ts`

**Benefits**: 80%+ cache hit rate, reduced database load, faster response times

### Performance Targets & Monitoring

**Required Targets:**

| Metric                | Target | Alert Threshold |
| --------------------- | ------ | --------------- |
| Database Query        | <50ms  | >750ms          |
| API Response          | <500ms | >1000ms         |
| Cache Hit Rate        | >80%   | <70%            |
| Page Load (FCP)       | <1.8s  | >3.0s           |
| Web Vitals LCP        | <2.5s  | >4.0s           |
| Web Vitals FID        | <100ms | >300ms          |
| Web Vitals CLS        | <0.1   | >0.25           |
| Bundle Size (Initial) | <250KB | >500KB          |

**Real-World Results Achieved:**

- Admin Metrics API: 70% faster (6 queries → 1 aggregation)
- Product List: 4x faster (4169ms → 500ms via parallel loading)
- Cache efficiency: 89% hit rate
- API calls after mutations: 75% reduction
- Database query times: 60% reduction (avg 7ms)
- **Performance Optimization (2025)**: 30% API reduction (323 → 225 calls)
- **Route-aware gating**: 100% elimination of unused dashboard calls
- **Shared context pattern**: 8x → 1x user preferences calls

---

## Quality Standards

### 1. Code Simplicity & Maintainability (MANDATORY)

> **Target Audience**: Write code that a developer with 2 years of experience
> can understand, maintain, and extend confidently.

#### Core Simplicity Principles

**1. Explicit Over Clever**

```typescript
// ✅ CORRECT: Simple and clear
function calculateDiscount(price: number, discountPercent: number): number {
  return price * (discountPercent / 100);
}

// ❌ WRONG: Clever but confusing
const calcDisc = (p: number, d: number) => p * (d / 100);
```

**2. Avoid Over-Abstraction**

```typescript
// ✅ CORRECT: Direct and simple
async function getProducts() {
  const products = await productService.getProducts();
  return products;
}

// ❌ WRONG: Unnecessary abstraction layers
async function getProducts() {
  return await getDataFactory()
    .createLoader('products')
    .withStrategy('default')
    .execute();
}
```

**3. Clear, Descriptive Naming**

```typescript
// ✅ CORRECT: Self-documenting
function validateUserEmailAddress(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// ❌ WRONG: Cryptic abbreviations
function valUEA(e: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e);
}
```

**4. Limit Nesting Depth** (Max 3 levels)

```typescript
// ✅ CORRECT: Early returns reduce nesting
async function processOrder(order: Order) {
  if (!order.items.length) {
    return { error: 'No items' };
  }

  if (!order.customerId) {
    return { error: 'No customer' };
  }

  return await createOrder(order);
}

// ❌ WRONG: Deep nesting (hard to follow)
async function processOrder(order: Order) {
  if (order.items.length) {
    if (order.customerId) {
      if (order.payment) {
        if (order.shipping) {
          return await createOrder(order);
        }
      }
    }
  }
}
```

**5. One Responsibility Per Function**

```typescript
// ✅ CORRECT: Single, clear purpose
function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(amount);
}

function calculateTotal(items: OrderItem[]): number {
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
}

// ❌ WRONG: Multiple responsibilities
function processOrderData(items: OrderItem[]): string {
  const total = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(total);
  logToAnalytics('order_processed', { total });
  sendEmailNotification(formatted);
  return formatted;
}
```

**6. Prefer Standard Patterns Over Custom Solutions**

```typescript
// ✅ CORRECT: Use established BaseService pattern
export class CustomerService extends BaseService {
  constructor() {
    super('CustomerService', '/api/customers');
  }

  async getCustomers(params: CustomerQuery): Promise<CustomerList> {
    return this.executeWithLogging('getCustomers', () =>
      this.get<CustomerList>(`?${this.buildQueryString(params)}`)
    );
  }
}

// ❌ WRONG: Custom error handling and logging
export class CustomerService {
  async getCustomers(params: CustomerQuery): Promise<CustomerList> {
    try {
      const start = Date.now();
      const result = await fetch(/* ... */);
      console.log(`Duration: ${Date.now() - start}ms`);
      return result;
    } catch (err) {
      console.error('Failed:', err);
      throw new CustomError(err);
    }
  }
}
```

**7. Avoid Magic Numbers and Strings**

```typescript
// ✅ CORRECT: Named constants
const STALE_TIME_MS = 30000; // 30 seconds
const CACHE_TIME_MS = 120000; // 2 minutes
const MAX_ITEMS_PER_PAGE = 20;

useQuery({
  queryKey: productKeys.list(params),
  queryFn: () => productService.getProducts(params),
  staleTime: STALE_TIME_MS,
  gcTime: CACHE_TIME_MS,
});

// ❌ WRONG: Unexplained magic numbers
useQuery({
  queryKey: productKeys.list(params),
  queryFn: () => productService.getProducts(params),
  staleTime: 30000, // What does this mean?
  gcTime: 120000, // Why this value?
});
```

**8. Document Complexity When Unavoidable**

```typescript
// ✅ CORRECT: Complex logic explained
/**
 * Calculates weighted average discount across product tiers.
 * Uses exponential decay for volume discounts (bulk purchases get higher discounts).
 *
 * @example
 * // 100 items with 10% base discount = 12.5% effective discount
 * calculateVolumeDiscount(100, 0.1) // Returns 0.125
 */
function calculateVolumeDiscount(
  quantity: number,
  baseDiscount: number
): number {
  const volumeMultiplier = Math.log10(quantity + 1) / 2;
  return baseDiscount * (1 + volumeMultiplier);
}

// ❌ WRONG: No explanation for complex logic
function calcVolDisc(q: number, d: number): number {
  return d * (1 + Math.log10(q + 1) / 2);
}
```

#### Simplicity Checklist (Required Before Commit)

- [ ] Can a 2-year developer understand this code in 5 minutes?
- [ ] Are all function names self-explanatory?
- [ ] Is nesting depth ≤ 3 levels?
- [ ] Are magic numbers/strings replaced with named constants?
- [ ] Does each function do one thing?
- [ ] Are established patterns used (BaseService, React Query, etc.)?
- [ ] Is complex logic documented with examples?
- [ ] Would this code pass a junior developer code review?

#### When Complexity Is Necessary

Sometimes complexity is unavoidable (performance optimization, security,
algorithms). In these cases:

1. **Isolate complexity** in dedicated utility functions
2. **Document thoroughly** with examples and rationale
3. **Add unit tests** to demonstrate expected behavior
4. **Provide usage examples** for common cases
5. **Consider adding to `docs/LESSONS_LEARNED.md`** if particularly tricky

**Example:**

```typescript
// ✅ CORRECT: Complex algorithm isolated and documented
/**
 * Implements Fisher-Yates shuffle for cryptographically secure randomization.
 * Required for security compliance in proposal team assignment.
 *
 * @see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 * @see docs/LESSONS_LEARNED.md#secure-randomization
 */
function secureRandomShuffle<T>(array: T[]): T[] {
  // Implementation with detailed comments
}

// Simple public interface
export function assignTeamRandomly(members: TeamMember[]): TeamMember[] {
  return secureRandomShuffle(members);
}
```

#### Anti-Patterns (Never Do This)

❌ **Deeply nested ternaries**

```typescript
// ❌ WRONG
const status = user
  ? user.active
    ? user.verified
      ? 'active'
      : 'unverified'
    : 'inactive'
  : 'guest';

// ✅ CORRECT
function getUserStatus(user?: User): string {
  if (!user) return 'guest';
  if (!user.active) return 'inactive';
  if (!user.verified) return 'unverified';
  return 'active';
}
```

❌ **Chained optional chaining**

```typescript
// ❌ WRONG (hard to debug)
const city = user?.profile?.address?.location?.city?.name;

// ✅ CORRECT (explicit with defaults)
function getUserCity(user: User): string {
  return user.profile?.address?.location?.city?.name ?? 'Unknown';
}
```

❌ **Clever regex without explanation**

```typescript
// ❌ WRONG
const isValid =
  /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(
    pwd
  );

// ✅ CORRECT
const PASSWORD_REQUIREMENTS = {
  minLength: 8,
  requireUppercase: true,
  requireLowercase: true,
  requireNumber: true,
  requireSpecial: true,
};

function validatePassword(password: string): boolean {
  // Clear validation with helpful error messages
  if (password.length < PASSWORD_REQUIREMENTS.minLength) return false;
  if (PASSWORD_REQUIREMENTS.requireUppercase && !/[A-Z]/.test(password))
    return false;
  // ... etc
  return true;
}
```

#### Benefits of Simple Code

1. **Faster onboarding** - New developers productive in days, not weeks
2. **Fewer bugs** - Simple code is easier to test and reason about
3. **Easier maintenance** - Changes are straightforward and low-risk
4. **Better collaboration** - Team members can review and contribute confidently
5. **Lower technical debt** - Less need for future refactoring

### 2. TypeScript Compliance (REQUIRED)

```bash
npm run type-check  # Must return 0 errors before commit
```

**Rules:**

- No `any` types (except where documented as necessary)
- Align types with Prisma schema
- Use Zod for runtime validation
- Export types from feature modules

### 3. Testing Requirements

```bash
npm test                          # Run test suite
npx tsx scripts/app-cli.ts --help # CLI testing
```

**Required Tests:**

- Unit tests for business logic
- Integration tests for API routes
- E2E tests for critical user flows
- Real database data (no mocks for main features)

### 4. Accessibility (WCAG 2.1 AA)

**Requirements:**

- 44px touch targets for mobile
- Keyboard navigation (Tab, Enter, Escape)
- Screen reader support (ARIA labels, roles)
- Color contrast ratios (4.5:1 for text)
- Focus states clearly visible

### 5. Security Requirements

**Environment Validation:**

```typescript
// src/env.mjs
export const env = createEnv({
  server: {
    DATABASE_URL: z.string().url(),
    NEXTAUTH_SECRET: z.string().min(32),
  },
  runtimeEnv: process.env,
});
```

**Security Headers** (next.config.js)

```javascript
headers: [
  { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
];
```

**Rate Limiting:**

- Authentication: 5 requests / 15min
- Admin endpoints: 20 requests / 1min
- API endpoints: 100 requests / 1min
- Cache operations: 10 requests / 1min

**Multi-Tenancy Isolation (MANDATORY)**

- Resolve the tenant ID from the authenticated user (or validated header) in the
  route handler, pass it into the service call, and execute work inside
  `runWithTenantContext({ tenantId })`. Treat `getCurrentTenant()` as a
  development-only fallback, never the primary source in APIs.
- API routes
  - `createRoute(...)` already exposes `user.tenantId`; use it to wrap the
    handler with `runWithTenantContext`.
  - NextRequest routes must call `withTenantContext(req, () => prisma...)` to
    establish the async context before touching the database.
- Prisma tenant middleware only injects `tenantId` for models with a dedicated
  column and blocks cross-tenant access. For relation tables (e.g.
  `proposal_products`) ensure the query carries an explicit tenant filter via
  related records (`product: { tenantId }`, etc.).
- Header enforcement: `x-tenant-id` must match the JWT tenant; mismatches are
  rejected before business logic runs.
- Create/update operations must set `tenantId` from the resolved tenant
  parameter (threaded from the route) so per-tenant uniques
  (`@@unique([tenantId, ...])`) remain valid.

### 6. Pre-Implementation Checklist

**Before Starting:**

- [ ] `npm run audit:duplicates` - Check existing patterns
- [ ] Review Prisma schema for field names
- [ ] Check existing API endpoints
- [ ] Plan feature structure (schemas, keys, hooks)
- [ ] Verify route boundaries needed

**Before Committing:**

- [ ] `npm run type-check` - 0 errors
- [ ] Route boundaries added (loading/error)
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No console.log statements

### 7. Component Design Standards (NEW COMPONENTS)

**Design for Reuse and Clarity** (MANDATORY):

**Props API:**

- Keep prop names explicit and consistent across the design system
- Use narrow, well-typed props; avoid broad `any` or ambiguous objects
  - Provide sensible defaults; mark truly required props

**Composition:**

- Prefer composition over inheritance
- Expose building blocks (subcomponents or render props) when useful

**Interop & Styling:**

- Support `className` pass-through and merge safely
  - Forward refs with `React.forwardRef` for focus/measure/accessibility
  - Avoid leaking internal DOM structure through fragile selectors

**Accessibility:**

- Keyboard navigation and ARIA roles/labels where applicable
- Meet WCAG 2.1 AA for focus states and contrast

**Performance:**

- Avoid creating new object/array literals in render paths
- Memoize expensive computations; stabilize callbacks
- Defer heavy logic until needed (lazy mount where applicable)

**Documentation & Tests:**

- Add concise JSDoc on components and props
- Include usage examples (or Storybook stories if available)
- Add basic unit/integration tests for critical behavior

**New Component Checklist:**

- [ ] Clear, minimal props API with defaults
- [ ] `className` passthrough and `forwardRef` implemented
- [ ] A11y coverage: roles, labels, keyboard, focus states
- [ ] Stable renders (no unnecessary re-renders)
- [ ] Examples/tests added alongside the component

---

## Platform Configuration

### Feature Flags (REQUIRED)

**Purpose**: Gate new features and experiments behind toggles.

```typescript
// Provider: src/components/providers/FlagsProvider.tsx
import { useFeatureFlag } from '@/hooks/useFeatureFlag';

// Usage in components
const MyComponent = () => {
  const isNewFeatureEnabled = useFeatureFlag('new-feature');

  if (!isNewFeatureEnabled) return <LegacyFeature />;
  return <NewFeature />;
};
```

**Configuration**:

- Default values from `getFeatureFlags()` (env-configured)
- Default safe-off in production
- Use for gradual rollouts and A/B testing

**Reference**: `src/components/providers/FlagsProvider.tsx`

### CORS & Middleware (REQUIRED)

**CORS Configuration** (`middleware.ts`):

```typescript
// ✅ CORRECT: CORS via CORS_ORIGINS environment variable
CORS_ORIGINS = 'https://app.example.com,https://admin.example.com';

// middleware.ts enforces:
// - Vary header for proper caching
// - Limited allowed headers
// - Origin validation
```

**Edge Middleware Features**:

- Security headers applied at edge
- Rate limiting integration
- `x-request-id` attached to every response
- PWA assets bypass heavy checks (`/sw.js`, `/icons/*`, `/manifest.json`)

**Reference**: `src/middleware.ts`

### Three-Tier Access Control System

> **Architecture**: PosalPro implements a unified three-tier access control
> system for comprehensive security.

#### Tier 1: Entitlements (Tenant Feature Licensing)

**Purpose**: Control features available to tenant based on subscription

**Use Cases**:

- Enterprise vs Standard features
- Module access control
- Capacity limits
- Add-on features

**Implementation**:

```typescript
// ✅ CORRECT: Use entitlements for admin/tenant features
export const GET = createRoute(
  {
    entitlements: ['feature.admin.services.read'],
  },
  async ({ user }) => {
    // Admin/premium feature logic
  }
);
```

**Storage**: `Entitlement` table (cached server-side) **Examples**:
`feature.admin.services.read`, `feature.analytics.enhanced`,
`feature.api.access`

#### Tier 2: Roles (User Grouping)

**Purpose**: Group users and define their general responsibilities

**Use Cases**:

- Define user types
- Establish hierarchy
- Simplify permission assignment

**Implementation**: `Role` model with hierarchy support (`parentId`)
**Storage**: `Role`, `UserRole`, `RolePermission` tables **Examples**: System
Administrator, Proposal Manager, Sales Representative

#### Tier 3: Permissions (Action-Level Control) - MANDATORY

**Purpose**: Granular control over specific actions on specific resources

**Use Cases**:

- CRUD operations
- Resource-specific access
- Fine-grained authorization

**Implementation** (REQUIRED for new routes):

```typescript
// ✅ CORRECT: Dynamic permissions from database
export const GET = createRoute({
  requireAuth: true,
  permissions: ['proposals:read'], // Checked dynamically
}, async ({ user }) => {
  // Your handler
});

// ❌ AVOID: Hard-coded entitlements for standard routes
export const GET = createRoute({
  entitlements: ['feature.proposals.read'], // Only for tenant licensing
}, ...);
```

**Format**: `<resource>:<action>` (e.g., `proposals:read`, `customers:create`)
**Scopes**: ALL, TENANT, DEPARTMENT, OWN **Storage**: `Permission`,
`RolePermission`, `UserPermission` tables

**Permission Service**:

- `src/lib/services/PermissionService.ts` - Central resolution
- User role checking
- Database permission resolution
- Administrator bypass logic
- Performance caching

**Admin UI**:

- `/admin/system` → Roles Tab (manage roles & hierarchy)
- `/admin/system` → Permissions Tab (CRUD & assign to roles)
- `/admin/system` → Entitlements Tab (tenant feature licensing)

**Detailed Documentation**:

- `docs/ACCESS_CONTROL_STRATEGY.md` - Architecture and strategy
- `docs/ACCESS_CONTROL_IMPLEMENTATION.md` - Usage guide and examples

**Acceptance Criteria**:

- [ ] New routes use `permissions` config (not hard-coded entitlements)
- [ ] Permissions follow `<resource>:<action>` naming convention
- [ ] Admin routes use `entitlements` for tenant feature gates only
- [ ] PermissionService used for dynamic checking
- [ ] 401/403 semantics consistent (401 = not authenticated, 403 = no
      permission)

### Seat/Subscription Guardrails (OPTIONAL)

**Purpose**: Server-side enforcement for seat limits and subscription tiers.

**Configuration** (environment variables):

```bash
SEAT_ENFORCEMENT=true           # Enable seat limit checking
SUBSCRIPTION_ENFORCEMENT=true   # Enable subscription tier checking
```

**Built-in to createRoute**:

- Server-side guardrails automatically integrated
- Admin roles bypass enforcement
- Subscription status checked via `subscriptionService`
- Seat availability via `EntitlementService`
- No route changes required

**When to Use**:

- Use only when product policy requires enforcement
- Typical use: SaaS pricing tiers, user limits, premium features
- Default: Disabled (enable per deployment environment)

**Example**:

```typescript
// Automatic enforcement when env vars set
export const POST = createRoute(
  {
    requireAuth: true,
    permissions: ['users:create'],
    // Seat checking happens automatically if SEAT_ENFORCEMENT=true
  },
  async ({ user }) => {
    // Your handler - seat check already performed
  }
);
```

---

## Reference Implementation

### File Location Matrix

| Pattern           | Location                           | Example                                      |
| ----------------- | ---------------------------------- | -------------------------------------------- |
| Feature Module    | `src/features/[domain]/`           | `src/features/products/`                     |
| Schemas           | `src/features/[domain]/schemas.ts` | `src/features/products/schemas.ts`           |
| Query Keys        | `src/features/[domain]/keys.ts`    | `src/features/products/keys.ts`              |
| React Query Hooks | `src/features/[domain]/hooks/`     | `src/features/products/hooks/useProducts.ts` |
| Frontend Service  | `src/services/`                    | `src/services/productService.ts`             |
| Database Service  | `src/lib/services/`                | `src/lib/services/productService.ts`         |
| API Routes        | `src/app/api/[resource]/`          | `src/app/api/products/route.ts`              |
| UI Components     | `src/components/[domain]/`         | `src/components/products/`                   |
| Zustand Stores    | `src/lib/store/`                   | `src/lib/store/productStore.ts`              |
| Route Pages       | `src/app/(dashboard)/[domain]/`    | `src/app/(dashboard)/products/page.tsx`      |

### Module Reference Implementations

**Gold Standard Modules** (follow these patterns):

- **Products**: Complete feature implementation
  - `src/features/products/` - Feature module
  - `src/services/productService.ts` - Frontend service
  - `src/lib/services/productService.ts` - Database service
  - `src/app/api/products/route.ts` - API routes
  - `src/app/(dashboard)/products/page.tsx` - UI with route boundaries

- **Customers**: CRUD operations + profile management
- **Proposals**: Multi-step wizard + state management

### Component Traceability Matrix

**User Stories → Components Mapping:**

- US-4.1 (Product Management) → ProductList, ProductForm, ProductDetail
- US-3.1 (Customer Management) → CustomerList, CustomerProfile
- US-2.1 (Proposal Creation) → ProposalWizard, ProposalSteps

**Hypothesis Validation:**

- H5 (Modern Data Fetching) → React Query patterns, cursor pagination
- H8 (Performance Optimization) → Caching, optimistic updates
- H11 (Error Handling) → ErrorHandlingService, StandardError

---

## Anti-Patterns & Pitfalls

### What NOT to Do

1. ❌ **Creating new APIs when existing ones work**
   - Check `npm run audit:duplicates` first
   - Review existing endpoints before creating new ones

2. ❌ **Storing server state in Zustand**
   - Use React Query for all server data
   - Zustand only for UI state (filters, selections, modals)

3. ❌ **Manual response envelope handling**
   - HTTP client unwraps automatically
   - Never manually check `response.ok`

4. ❌ **Long stale times for dynamic data**
   - Use 30s for user data, not 5min
   - Aggressive cache invalidation

5. ❌ **Dynamic component IDs with random values**
   - Stable IDs for testing and accessibility
   - Use entity IDs, not `Math.random()`

6. ❌ **Console.log in production**
   - Use structured logger: `logInfo`, `logError`, `logDebug`
   - Automatic request-ID correlation

7. ❌ **Composite hooks creating objects on every render**
   - Memoize objects and arrays
   - Use individual selectors

8. ❌ **Skipping route boundaries**
   - Every route needs `loading.tsx` and `error.tsx`
   - Required for proper error handling and UX

9. ❌ **Field name mismatches**
   - Database schema is source of truth
   - Check Prisma schema before creating types

10. ❌ **Direct Prisma imports in API routes**
    - Delegate to database services
    - API routes are thin boundaries only

11. ❌ **Loading dashboard data on all pages**
    - Use route-aware gating with `enabled` option
    - Conditional loading based on pathname

12. ❌ **Multiple instances of same React Query hook**
    - Use shared context providers for common data
    - Prevents duplicate API calls across components

### Common Mistakes

**Issue**: Update mutation triggers extra API call

```typescript
// ❌ WRONG
onSuccess: async (data, variables) => {
  const fresh = await service.get(variables.id); // Extra API call
  queryClient.setQueryData(keys.byId(variables.id), fresh);
};

// ✅ CORRECT
onSuccess: (data, variables) => {
  queryClient.setQueryData(keys.byId(variables.id), data); // Use mutation response
};
```

**Issue**: Infinite re-render loops

```typescript
// ❌ WRONG: Unstable dependencies
useEffect(() => {
  fetchData();
}, [apiClient, handleError]); // Functions recreated on every render

// ✅ CORRECT: Empty array for mount-only effects
useEffect(() => {
  fetchData();
}, []); // eslint-disable-line react-hooks/exhaustive-deps
```

**Issue**: Schema validation inconsistencies

```typescript
// ❌ WRONG: Update schema missing .nullable()
export const CustomerSchema = z.object({
  phone: z.string().nullable().optional(),
});
export const CustomerUpdateSchema = z.object({
  phone: z.string().optional(), // Missing .nullable()
});

// ✅ CORRECT: Consistent patterns
export const CustomerUpdateSchema = z.object({
  phone: z.string().nullable().optional(), // Matches main schema
});
```

### Migration Lessons

**Lesson: Always use BaseService for frontend services**

- Provides automatic error handling
- Structured logging with correlation
- Performance tracking
- Consistent patterns

**Lesson: Cursor pagination is mandatory**

- Offset pagination deprecated
- Better performance on large datasets
- Consistent across all modules

**Lesson: Route boundaries prevent errors**

- Every route needs loading.tsx and error.tsx
- Improves UX during loading and errors
- Proper error recovery mechanisms

**Lesson: Optimistic updates reduce API calls**

- Use mutation response directly
- Single targeted invalidation
- 75% reduction in cache operations

**Lesson: Route-aware data fetching eliminates unnecessary calls**

- Use `enabled` option based on pathname
- Dashboard APIs only load on relevant routes
- 100% elimination of unused API calls

**Lesson: Shared context providers prevent duplicate queries**

- Hoist common React Query hooks to providers
- Components use context instead of direct hooks
- 8x → 1x reduction in duplicate API calls

---

## Configuration Reference

### Environment Variables

**Required** (src/env.mjs, src/env.ts):

```typescript
DATABASE_URL: z.string().url();
NEXTAUTH_SECRET: z.string().min(32);
NEXTAUTH_URL: z.string().url();
NODE_ENV: z.enum(['development', 'test', 'production']);
```

### React Query Defaults

**QueryClient Configuration:**

```typescript
{
  defaultOptions: {
    queries: {
      staleTime: 30000,              // 30s
      gcTime: 120000,                // 2min
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      retry: 2,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 5000),
    },
  },
}
```

### Cache TTLs

| Data Type       | TTL        | Use Case             |
| --------------- | ---------- | -------------------- |
| Session & Auth  | 180-300s   | Security-focused     |
| User Data       | 900-1200s  | Moderate changes     |
| API Responses   | 300s       | Balanced performance |
| Product Catalog | 1800s      | Relatively static    |
| Cache Warming   | 3600-7200s | Prefetched data      |

### Performance Thresholds

| Metric         | Target | Alert Threshold |
| -------------- | ------ | --------------- |
| Database Query | <50ms  | >750ms          |
| API Response   | <500ms | >1000ms         |
| Cache Hit Rate | >80%   | <70%            |
| Web Vitals LCP | <2.5s  | >4.0s           |
| Web Vitals FID | <100ms | >300ms          |
| Web Vitals CLS | <0.1   | >0.25           |

### Security Configuration

**Rate Limits:**

- Auth endpoints: 5 req/15min
- Admin endpoints: 20 req/min
- API endpoints: 100 req/min
- Cache operations: 10 req/min

**Headers** (next.config.js):

```javascript
{ key: 'X-Frame-Options', value: 'SAMEORIGIN' }
{ key: 'X-Content-Type-Options', value: 'nosniff' }
{ key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' }
{ key: 'Permissions-Policy', value: 'camera=(), microphone=()' }
```

---

## Related Documents

**Must Read:**

- `docs/PROJECT_REFERENCE.md` - Architecture and API documentation
- `docs/DEVELOPMENT_STANDARDS.md` - Code quality and patterns
- `docs/WIREFRAME_INTEGRATION_GUIDE.md` - UI/UX implementation

**Implementation Tracking:**

- `docs/IMPLEMENTATION_LOG.md` - After every implementation
- `docs/LESSONS_LEARNED.md` - Complex insights and patterns
- `docs/USER_STORY_TRACEABILITY_MATRIX.md` - Story → Component mapping

**Specialized Guides:**

- `docs/ACCESSIBILITY_SPECIFICATION.md` - WCAG 2.1 AA compliance
- `docs/DATA_MODEL.md` - Database schema and relationships
- `docs/COMPONENT_STRUCTURE.md` - Component patterns

---

## How to Use This Document

**This is your source of truth for all PosalPro MVP2 implementations.**

### Quick Reference Flow

1. **Check existing implementations first** (`npm run audit:duplicates`)
2. **Follow Feature-Based Architecture** (`src/features/[domain]/`)
3. **Database-First Design** (check Prisma schema before creating types)
4. **Implement Route Boundaries** (loading.tsx and error.tsx for all routes)
5. **Use modern patterns** (React Query + Zustand + BaseService)
6. **Use structured logging** (`@/lib/logger`) with automatic request-ID
   correlation
7. **Reference existing feature modules** (Products module is gold standard)

### When to Reference Specific Sections

**Starting New Feature:**

- Read: [Quick Start Guide](#quick-start-guide)
- Check: [Reference Implementation](#reference-implementation)
- Follow: [Mandatory Patterns](#mandatory-patterns)

**Creating API Route:**

- Review: [API & Routes](#api--routes)
- Check: [Platform Configuration](#platform-configuration) (permissions)
- Reference: Existing routes in `src/app/api/products/route.ts`

**Performance Issue:**

- Check: [Performance & Optimization](#performance--optimization)
- Review: Anti-patterns section
- Common patterns: Parallel loading, optimistic updates, caching

**Build/Type Error:**

- Review: [Quality Standards](#quality-standards)
- Check: [Anti-Patterns & Pitfalls](#anti-patterns--pitfalls)
- Run: `npm run type-check` (must be 0 errors)

**Creating Component:**

- Review: Component Design Standards (Quality Standards section)
- Check: Accessibility requirements
- Reference: `src/components/products/` for patterns

### Key Principles

**Code Simplicity First (2-Year Developer Test):**

- Write code a 2-year developer can understand in 5 minutes
- Explicit over clever; simple over smart
- Max 3 levels of nesting; one responsibility per function
- Named constants instead of magic numbers
- Document complexity when unavoidable

**Feature-First Organization:**

- All domain logic in `src/features/[domain]/`
- Schemas centralized in `schemas.ts`
- Query keys in `keys.ts`
- React Query hooks in `hooks/`

**Database Schema is Source of Truth:**

- Always check `prisma/schema.prisma` first
- Field names must match exactly
- Align Zod schemas with Prisma schema

**BaseService for All Frontend Services:**

- Extend `BaseService` for automatic error handling
- Use `executeWithLogging()` for operations
- Structured logging with performance tracking

**Route Boundaries for All Routes:**

- Every route needs `loading.tsx` and `error.tsx`
- Proper loading states improve UX
- Error boundaries enable recovery

**When in Doubt:**

- Check existing implementations first
- Follow Products module patterns
- Use BaseService pattern
- Ask: "Can a junior dev understand this?"
- Consult Component Traceability Matrix
- Ask: "Does existing code do this?"

### Attach This Document

**Attach CORE_REQUIREMENTS.md to every Cursor prompt** when working on new
features to ensure consistency with established patterns.

---

**Document Version**: 2.0 (Modernized) **Last Updated**: January 2025 **Next
Review**: As needed based on architectural changes
