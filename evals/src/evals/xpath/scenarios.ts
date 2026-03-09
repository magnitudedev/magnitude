/**
 * XPath 3.1 eval scenarios
 *
 * Tests LLM ability to write correct XPath 3.1 queries, covering:
 * - Pure XML navigation (attributes, predicates, axes)
 * - JSON querying via parse-json() and ? lookup operator
 * - Composed XML + JSON (drill through XML to JSON body content)
 *
 * Each scenario provides XML/JSON data, a natural language question,
 * and an expected result. The LLM must respond with a valid XPath 3.1
 * expression that produces the expected output when evaluated.
 */

export interface XPathScenario {
  id: string
  description: string
  /** XML document string (parsed with slimdom), or null if pure variable-based */
  xml: string | null
  /** Variables to bind (for JSON-only scenarios) */
  variables?: Record<string, unknown>
  /** Natural language description of what to extract */
  question: string
  /** Expected result — string, number, boolean, or array of primitives */
  expected: unknown
  /** Whether the result is a sequence (array) vs single value */
  isSequence?: boolean
}

// =============================================================================
// Tier 1: Pure XML — basic XPath that every LLM should know
// =============================================================================

export const xmlBasicAttr: XPathScenario = {
  id: 'xml/basic-attr',
  description: 'Extract a single attribute value from an XML element',
  xml: `<library>
  <book id="1" title="Dune" author="Frank Herbert" year="1965"/>
  <book id="2" title="Neuromancer" author="William Gibson" year="1984"/>
  <book id="3" title="Snow Crash" author="Neal Stephenson" year="1992"/>
</library>`,
  question: 'Get the title of the book with id="2".',
  expected: 'Neuromancer',
}

export const xmlPredicateFilter: XPathScenario = {
  id: 'xml/predicate-filter',
  description: 'Filter elements using a numeric predicate on an attribute',
  xml: `<inventory>
  <item name="Widget" price="9.99" stock="150"/>
  <item name="Gadget" price="24.95" stock="42"/>
  <item name="Doohickey" price="4.50" stock="300"/>
  <item name="Thingamajig" price="15.00" stock="0"/>
</inventory>`,
  question: 'Get the names of all items where the price is greater than 10.',
  expected: ['Gadget', 'Thingamajig'],
  isSequence: true,
}

export const xmlNestedElements: XPathScenario = {
  id: 'xml/nested-elements',
  description: 'Navigate nested XML elements to extract text content',
  xml: `<company>
  <department name="Engineering">
    <team name="Backend">
      <member>Alice</member>
      <member>Bob</member>
    </team>
    <team name="Frontend">
      <member>Carol</member>
    </team>
  </department>
  <department name="Marketing">
    <team name="Growth">
      <member>Dave</member>
      <member>Eve</member>
    </team>
  </department>
</company>`,
  question: 'Get the names of all members in the Backend team.',
  expected: ['Alice', 'Bob'],
  isSequence: true,
}

export const xmlMultiPredicate: XPathScenario = {
  id: 'xml/multi-predicate',
  description: 'Filter using multiple predicates on different attributes',
  xml: `<catalog>
  <product category="electronics" brand="Acme" rating="4.5" inStock="true"/>
  <product category="electronics" brand="Globex" rating="3.8" inStock="false"/>
  <product category="books" brand="Acme" rating="4.9" inStock="true"/>
  <product category="electronics" brand="Acme" rating="4.1" inStock="false"/>
</catalog>`,
  question: 'Get the rating of electronics products from brand "Acme" that are in stock.',
  expected: '4.5',
}

export const xmlPositionalAndCount: XPathScenario = {
  id: 'xml/positional-and-count',
  description: 'Use positional access and count() function',
  xml: `<playlist>
  <track title="Song A" duration="180"/>
  <track title="Song B" duration="240"/>
  <track title="Song C" duration="200"/>
  <track title="Song D" duration="310"/>
  <track title="Song E" duration="195"/>
</playlist>`,
  question: 'How many tracks are in the playlist?',
  expected: '5',
}

// =============================================================================
// Tier 2: Pure JSON via variables — XPath 3.1 map/array syntax
// =============================================================================

export const jsonBasicLookup: XPathScenario = {
  id: 'json/basic-lookup',
  description: 'Look up a field in a JSON object using ? operator',
  xml: null,
  variables: {
    data: { name: 'Alice', age: 30, city: 'Seattle' },
  },
  question: 'The variable $data contains a JSON object. Get the value of the "city" field.',
  expected: 'Seattle',
}

export const jsonArrayFilter: XPathScenario = {
  id: 'json/array-filter',
  description: 'Filter an array of objects using a predicate on a field',
  xml: null,
  variables: {
    data: {
      users: [
        { name: 'Alice', role: 'admin', active: true },
        { name: 'Bob', role: 'user', active: true },
        { name: 'Carol', role: 'admin', active: false },
        { name: 'Dave', role: 'user', active: false },
      ],
    },
  },
  question: 'The variable $data contains a JSON object. Get the names of all active admin users.',
  expected: ['Alice'],
  isSequence: true,
}

export const jsonNestedAccess: XPathScenario = {
  id: 'json/nested-access',
  description: 'Navigate deeply nested JSON structures',
  xml: null,
  variables: {
    data: {
      response: {
        metadata: { page: 1, totalPages: 5 },
        results: [
          { id: 101, title: 'First Post', tags: ['news', 'featured'] },
          { id: 102, title: 'Second Post', tags: ['tutorial'] },
          { id: 103, title: 'Third Post', tags: ['news'] },
        ],
      },
    },
  },
  question: 'The variable $data contains a JSON object. Get the title of the result with id 102.',
  expected: 'Second Post',
}

export const jsonNumericComparison: XPathScenario = {
  id: 'json/numeric-comparison',
  description: 'Filter JSON array using numeric comparison on a field',
  xml: null,
  variables: {
    data: {
      transactions: [
        { id: 'tx1', amount: 150.00, currency: 'USD' },
        { id: 'tx2', amount: 3200.00, currency: 'USD' },
        { id: 'tx3', amount: 89.99, currency: 'EUR' },
        { id: 'tx4', amount: 500.00, currency: 'USD' },
      ],
    },
  },
  question: 'The variable $data contains a JSON object. Get the ids of all USD transactions with amount greater than 200.',
  expected: ['tx2', 'tx4'],
  isSequence: true,
}

export const jsonMultiplePredicates: XPathScenario = {
  id: 'json/multiple-predicates',
  description: 'Apply multiple filter predicates on a JSON array',
  xml: null,
  variables: {
    data: {
      employees: [
        { name: 'Alice', department: 'Engineering', level: 'Senior', salary: 150000 },
        { name: 'Bob', department: 'Engineering', level: 'Junior', salary: 85000 },
        { name: 'Carol', department: 'Marketing', level: 'Senior', salary: 130000 },
        { name: 'Dave', department: 'Engineering', level: 'Senior', salary: 160000 },
        { name: 'Eve', department: 'Marketing', level: 'Junior', salary: 70000 },
      ],
    },
  },
  question: 'The variable $data contains a JSON object. Get the names of senior engineers with salary above 140000.',
  expected: ['Alice', 'Dave'],
  isSequence: true,
}

// =============================================================================
// Tier 3: XML + embedded JSON — compose / and ? across boundaries
// =============================================================================

export const composedSimple: XPathScenario = {
  id: 'composed/simple',
  description: 'Navigate XML to find a JSON body, then extract a field',
  xml: `<api>
  <response status="200" endpoint="/user">
    <body>{"id": 42, "username": "jdoe", "email": "jdoe@example.com"}</body>
  </response>
</api>`,
  question: 'The <body> element contains JSON. Get the email address from the response.',
  expected: 'jdoe@example.com',
}

export const composedFilterArray: XPathScenario = {
  id: 'composed/filter-array',
  description: 'Navigate XML to JSON body, then filter an array within it',
  xml: `<services>
  <service name="auth" version="2.1">
    <config>{"endpoints": [{"path": "/login", "method": "POST", "public": true}, {"path": "/logout", "method": "POST", "public": false}, {"path": "/refresh", "method": "GET", "public": false}]}</config>
  </service>
  <service name="api" version="3.0">
    <config>{"endpoints": [{"path": "/users", "method": "GET", "public": true}, {"path": "/admin", "method": "GET", "public": false}]}</config>
  </service>
</services>`,
  question: 'The <config> elements contain JSON. Get the paths of all public endpoints in the "auth" service.',
  expected: ['/login'],
  isSequence: true,
}

export const composedMultipleJsonBodies: XPathScenario = {
  id: 'composed/multiple-json-bodies',
  description: 'Select between multiple XML elements with JSON bodies based on XML attributes',
  xml: `<results>
  <query id="q1" status="success">
    <data>{"rows": [{"col1": "a", "col2": 10}, {"col1": "b", "col2": 20}]}</data>
  </query>
  <query id="q2" status="error">
    <data>{"error": "timeout", "code": 504}</data>
  </query>
  <query id="q3" status="success">
    <data>{"rows": [{"col1": "x", "col2": 99}]}</data>
  </query>
</results>`,
  question: 'The <data> elements contain JSON. From the successful query with id="q1", get all col2 values.',
  expected: [10, 20],
  isSequence: true,
}

export const composedDeepNesting: XPathScenario = {
  id: 'composed/deep-nesting',
  description: 'Deep XML navigation followed by deep JSON navigation',
  xml: `<cloud>
  <region name="us-east-1">
    <cluster name="prod">
      <metrics>{"cpu": {"avg": 45.2, "max": 92.1, "min": 3.5}, "memory": {"avg": 68.0, "max": 95.4, "min": 12.1}, "requests": {"total": 150000, "errors": 42}}</metrics>
    </cluster>
    <cluster name="staging">
      <metrics>{"cpu": {"avg": 12.5, "max": 30.0, "min": 1.0}, "memory": {"avg": 25.0, "max": 50.0, "min": 5.0}, "requests": {"total": 500, "errors": 2}}</metrics>
    </cluster>
  </region>
</cloud>`,
  question: 'The <metrics> elements contain JSON. Get the max CPU value for the prod cluster in us-east-1.',
  expected: '92.1',
}

// =============================================================================
// All scenarios grouped by tier
// =============================================================================

export const ALL_SCENARIOS: XPathScenario[] = [
  // Tier 1: Pure XML
  xmlBasicAttr,
  xmlPredicateFilter,
  xmlNestedElements,
  xmlMultiPredicate,
  xmlPositionalAndCount,
  // Tier 2: Pure JSON
  jsonBasicLookup,
  jsonArrayFilter,
  jsonNestedAccess,
  jsonNumericComparison,
  jsonMultiplePredicates,
  // Tier 3: Composed XML + JSON
  composedSimple,
  composedFilterArray,
  composedMultipleJsonBodies,
  composedDeepNesting,
]
