// ============================================================================
// CORE LANGUAGE FUNDAMENTALS
// ============================================================================

// 1. Execution Contexts and Call Stack
function first() {
  console.log('First');
  second();
  console.log('First again');
}
function second() {
  console.log('Second');
  third();
}
function third() {
  console.log('Third');
}
// first(); // Call stack: first -> second -> third -> second -> first

// 2. Hoisting
console.log(x); // undefined (var is hoisted but not initialized)
var x = 5;

// console.log(y); // ReferenceError (temporal dead zone)
let y = 10;

hoistedFunc(); // Works! Function declarations are fully hoisted
function hoistedFunc() {
  console.log('I am hoisted');
}

// notHoisted(); // TypeError: notHoisted is not a function
var notHoisted = function() {
  console.log('Not hoisted');
};

// 3. Scope Chain and Lexical Scoping
const globalVar = 'global';

function outer() {
  const outerVar = 'outer';
  
  function inner() {
    const innerVar = 'inner';
    console.log(globalVar, outerVar, innerVar); // Has access to all
  }
  
  inner();
  // console.log(innerVar); // ReferenceError
}

// 4. Closures
function createCounter() {
  let count = 0;
  
  return {
    increment: () => ++count,
    decrement: () => --count,
    getCount: () => count
  };
}

const counter = createCounter();
console.log(counter.increment()); // 1
console.log(counter.increment()); // 2
console.log(counter.getCount()); // 2
// count variable is private and enclosed

// Practical closure example - data privacy
function bankAccount(initialBalance) {
  let balance = initialBalance;
  
  return {
    deposit: (amount) => {
      balance += amount;
      return balance;
    },
    withdraw: (amount) => {
      if (amount > balance) return 'Insufficient funds';
      balance -= amount;
      return balance;
    },
    getBalance: () => balance
  };
}

// 5. 'this' keyword in different contexts
const obj = {
  name: 'Object',
  regular: function() {
    console.log(this.name); // 'Object'
  },
  arrow: () => {
    console.log(this.name); // undefined (lexical this from outer scope)
  },
  nested: function() {
    const inner = () => {
      console.log(this.name); // 'Object' (arrow function inherits this)
    };
    inner();
  }
};

// Explicit binding
function greet(greeting) {
  console.log(`${greeting}, ${this.name}`);
}
const person = { name: 'Alice' };
greet.call(person, 'Hello'); // Hello, Alice
greet.apply(person, ['Hi']); // Hi, Alice
const boundGreet = greet.bind(person);
boundGreet('Hey'); // Hey, Alice

// Constructor context
function Person(name) {
  this.name = name;
  this.sayName = function() {
    console.log(this.name);
  };
}
const alice = new Person('Alice');

// 6. Prototypal Inheritance
function Animal(name) {
  this.name = name;
}
Animal.prototype.speak = function() {
  console.log(`${this.name} makes a sound`);
};

function Dog(name, breed) {
  Animal.call(this, name); // Call parent constructor
  this.breed = breed;
}

// Set up inheritance
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;

Dog.prototype.bark = function() {
  console.log(`${this.name} barks`);
};

const dog = new Dog('Buddy', 'Golden Retriever');
dog.speak(); // Buddy makes a sound
dog.bark(); // Buddy barks

// 7. Class Syntax (ES6)
class Vehicle {
  constructor(brand) {
    this.brand = brand;
  }
  
  drive() {
    console.log(`${this.brand} is driving`);
  }
  
  static compare(v1, v2) {
    return v1.brand === v2.brand;
  }
}

class Car extends Vehicle {
  constructor(brand, model) {
    super(brand);
    this.model = model;
  }
  
  drive() {
    super.drive();
    console.log(`Model: ${this.model}`);
  }
}

// 8. Object.create()
const protoObj = {
  greet() {
    console.log(`Hello, ${this.name}`);
  }
};

const newObj = Object.create(protoObj);
newObj.name = 'Bob';
newObj.greet(); // Hello, Bob

// ============================================================================
// ASYNCHRONOUS JAVASCRIPT
// ============================================================================

// 9. Event Loop Demonstration
console.log('1 - Sync');

setTimeout(() => console.log('2 - Macro task'), 0);

Promise.resolve().then(() => console.log('3 - Micro task'));

console.log('4 - Sync');
// Output: 1, 4, 3, 2 (microtasks execute before macrotasks)

// 10. Promises - Creation and Chaining
const myPromise = new Promise((resolve, reject) => {
  const success = true;
  setTimeout(() => {
    if (success) resolve('Success!');
    else reject('Failed!');
  }, 1000);
});

myPromise
  .then(result => {
    console.log(result);
    return result + ' More data';
  })
  .then(result => console.log(result))
  .catch(error => console.error(error))
  .finally(() => console.log('Cleanup'));

// 11. Promise Static Methods
const p1 = Promise.resolve(1);
const p2 = Promise.resolve(2);
const p3 = Promise.reject('Error');
const p4 = new Promise(resolve => setTimeout(() => resolve(4), 100));

// Promise.all - waits for all, fails if any fails
Promise.all([p1, p2, p4])
  .then(results => console.log('All:', results)); // [1, 2, 4]

// Promise.race - returns first settled promise
Promise.race([p1, p4])
  .then(result => console.log('Race:', result)); // 1

// Promise.allSettled - waits for all, returns all results
Promise.allSettled([p1, p2, p3])
  .then(results => console.log('AllSettled:', results));
  // [{status: 'fulfilled', value: 1}, {status: 'fulfilled', value: 2}, {status: 'rejected', reason: 'Error'}]

// Promise.any - returns first fulfilled promise
Promise.any([p3, p1, p2])
  .then(result => console.log('Any:', result)); // 1

// 12. Async/Await
async function fetchData() {
  try {
    const response = await fetch('https://api.example.com/data');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Parallel async operations
async function fetchMultiple() {
  const [users, posts] = await Promise.all([
    fetch('/api/users').then(r => r.json()),
    fetch('/api/posts').then(r => r.json())
  ]);
  return { users, posts };
}

// 13. Microtasks vs Macrotasks
console.log('Script start');

setTimeout(() => console.log('setTimeout'), 0);

Promise.resolve()
  .then(() => console.log('Promise 1'))
  .then(() => console.log('Promise 2'));

queueMicrotask(() => console.log('queueMicrotask'));

console.log('Script end');
// Output: Script start, Script end, Promise 1, queueMicrotask, Promise 2, setTimeout

// ============================================================================
// ADVANCED FUNCTIONS
// ============================================================================

// 14. Higher-Order Functions
const numbers = [1, 2, 3, 4, 5];

// Map - transform each element
const doubled = numbers.map(n => n * 2);

// Filter - select elements
const evens = numbers.filter(n => n % 2 === 0);

// Reduce - accumulate to single value
const sum = numbers.reduce((acc, n) => acc + n, 0);

// Custom higher-order function
function repeat(n, action) {
  for (let i = 0; i < n; i++) {
    action(i);
  }
}
repeat(3, i => console.log(`Iteration ${i}`));

// 15. Currying
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    } else {
      return function(...nextArgs) {
        return curried.apply(this, args.concat(nextArgs));
      };
    }
  };
}

function add(a, b, c) {
  return a + b + c;
}

const curriedAdd = curry(add);
console.log(curriedAdd(1)(2)(3)); // 6
console.log(curriedAdd(1, 2)(3)); // 6
console.log(curriedAdd(1)(2, 3)); // 6

// 16. Function Composition
const compose = (...fns) => x => fns.reduceRight((acc, fn) => fn(acc), x);
const pipe = (...fns) => x => fns.reduce((acc, fn) => fn(acc), x);

const addOne = x => x + 1;
const double = x => x * 2;
const square = x => x * x;

const composed = compose(square, double, addOne);
console.log(composed(3)); // ((3 + 1) * 2)^2 = 64

const piped = pipe(addOne, double, square);
console.log(piped(3)); // ((3 + 1) * 2)^2 = 64

// 17. Debounce Implementation
function debounce(fn, delay) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

const debouncedSearch = debounce((query) => {
  console.log('Searching for:', query);
}, 300);

// 18. Throttle Implementation
function throttle(fn, limit) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      fn.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

const throttledScroll = throttle(() => {
  console.log('Scroll event');
}, 1000);

// 19. Memoization
function memoize(fn) {
  const cache = new Map();
  return function(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      console.log('Cache hit');
      return cache.get(key);
    }
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
}

const fibonacci = memoize(function(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
});

console.log(fibonacci(40)); // Much faster with memoization

// 20. Generator Functions
function* numberGenerator() {
  let num = 0;
  while (true) {
    yield num++;
  }
}

const gen = numberGenerator();
console.log(gen.next().value); // 0
console.log(gen.next().value); // 1

function* asyncGenerator() {
  yield Promise.resolve(1);
  yield Promise.resolve(2);
  yield Promise.resolve(3);
}

// Generator for iteration
function* range(start, end) {
  for (let i = start; i <= end; i++) {
    yield i;
  }
}

for (let num of range(1, 5)) {
  console.log(num); // 1, 2, 3, 4, 5
}

// 21. IIFE (Immediately Invoked Function Expression)
(function() {
  const privateVar = 'I am private';
  console.log(privateVar);
})();

// Module pattern with IIFE
const myModule = (function() {
  let privateCounter = 0;
  
  return {
    increment: () => ++privateCounter,
    getCount: () => privateCounter
  };
})();

// ============================================================================
// OBJECTS AND DATA STRUCTURES
// ============================================================================

// 22. Object Property Descriptors
const obj1 = {};
Object.defineProperty(obj1, 'readOnly', {
  value: 42,
  writable: false,
  enumerable: true,
  configurable: false
});

// obj1.readOnly = 100; // Fails silently in non-strict mode
console.log(obj1.readOnly); // 42

Object.defineProperty(obj1, 'computed', {
  get() {
    return this._computed || 0;
  },
  set(value) {
    this._computed = value * 2;
  },
  enumerable: true
});

obj1.computed = 10;
console.log(obj1.computed); // 20

// 23. Getters and Setters
class Circle {
  constructor(radius) {
    this._radius = radius;
  }
  
  get radius() {
    return this._radius;
  }
  
  set radius(value) {
    if (value < 0) throw new Error('Radius cannot be negative');
    this._radius = value;
  }
  
  get area() {
    return Math.PI * this._radius ** 2;
  }
}

// 24. Object Methods
const obj2 = { a: 1, b: 2 };

// Freeze - makes object immutable
Object.freeze(obj2);
// obj2.a = 100; // Fails silently

// Seal - prevents adding/removing properties
const obj3 = { x: 1 };
Object.seal(obj3);
obj3.x = 100; // OK
// obj3.y = 200; // Fails silently

// preventExtensions - prevents adding new properties
const obj4 = { m: 1 };
Object.preventExtensions(obj4);
// obj4.n = 2; // Fails silently

// 25. Maps vs Objects
const map = new Map();
map.set('key1', 'value1');
map.set({ id: 1 }, 'object key'); // Objects as keys!
map.set(1, 'number key');

console.log(map.size); // 3
console.log(map.get('key1')); // value1
console.log(map.has('key1')); // true
map.delete('key1');

// Iteration
for (let [key, value] of map) {
  console.log(key, value);
}

// 26. Sets and WeakSets
const set = new Set([1, 2, 3, 3, 3]);
console.log(set.size); // 3 (duplicates removed)
set.add(4);
set.delete(1);
console.log(set.has(2)); // true

// WeakSet - holds weak references, allows garbage collection
const weakSet = new WeakSet();
let obj5 = { id: 1 };
weakSet.add(obj5);
// obj5 = null; // Can be garbage collected

// 27. WeakMaps
const weakMap = new WeakMap();
let keyObj = { id: 1 };
weakMap.set(keyObj, 'metadata');
console.log(weakMap.get(keyObj)); // metadata
// keyObj = null; // Can be garbage collected

// Use case: private data
const privateData = new WeakMap();

class Person {
  constructor(name, ssn) {
    this.name = name;
    privateData.set(this, { ssn });
  }
  
  getSSN() {
    return privateData.get(this).ssn;
  }
}

// 28. Symbols
const sym1 = Symbol('description');
const sym2 = Symbol('description');
console.log(sym1 === sym2); // false (symbols are unique)

// Use in objects - hidden properties
const SECRET_KEY = Symbol('secret');
const obj6 = {
  [SECRET_KEY]: 'hidden value',
  public: 'visible'
};

console.log(Object.keys(obj6)); // ['public'] - symbol key not enumerated
console.log(obj6[SECRET_KEY]); // 'hidden value'

// Well-known symbols
const iterableObj = {
  data: [1, 2, 3],
  [Symbol.iterator]() {
    let index = 0;
    return {
      next: () => ({
        value: this.data[index++],
        done: index > this.data.length
      })
    };
  }
};

for (let value of iterableObj) {
  console.log(value); // 1, 2, 3
}

// 29. Proxy and Reflect
const target = { name: 'John', age: 30 };

const handler = {
  get(target, prop) {
    console.log(`Getting ${prop}`);
    return Reflect.get(target, prop);
  },
  set(target, prop, value) {
    console.log(`Setting ${prop} to ${value}`);
    if (prop === 'age' && typeof value !== 'number') {
      throw new TypeError('Age must be a number');
    }
    return Reflect.set(target, prop, value);
  },
  has(target, prop) {
    console.log(`Checking if ${prop} exists`);
    return Reflect.has(target, prop);
  }
};

const proxy = new Proxy(target, handler);
console.log(proxy.name); // Getting name, John
proxy.age = 31; // Setting age to 31

// Validation proxy
function createValidatedObject(target, validators) {
  return new Proxy(target, {
    set(obj, prop, value) {
      if (validators[prop]) {
        validators[prop](value);
      }
      return Reflect.set(obj, prop, value);
    }
  });
}

// ============================================================================
// ES6+ FEATURES
// ============================================================================

// 30. Destructuring
const arr = [1, 2, 3, 4, 5];
const [first, second, ...rest] = arr;
console.log(first, second, rest); // 1, 2, [3, 4, 5]

const obj7 = { x: 1, y: 2, z: 3 };
const { x, y, ...others } = obj7;
console.log(x, y, others); // 1, 2, { z: 3 }

// Nested destructuring
const data = {
  user: {
    name: 'Alice',
    address: { city: 'NYC', zip: '10001' }
  }
};
const { user: { name, address: { city } } } = data;
console.log(name, city); // Alice, NYC

// Default values
const { a = 10, b = 20 } = { a: 5 };
console.log(a, b); // 5, 20

// 31. Spread and Rest
// Spread in arrays
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const combined = [...arr1, ...arr2];

// Spread in objects
const obj8 = { a: 1, b: 2 };
const obj9 = { c: 3, ...obj8, d: 4 };

// Rest parameters
function sum(...numbers) {
  return numbers.reduce((acc, n) => acc + n, 0);
}
console.log(sum(1, 2, 3, 4)); // 10

// 32. Template Literals
const name = 'Alice';
const age = 30;
const message = `Hello, ${name}! You are ${age} years old.`;

// Tagged templates
function highlight(strings, ...values) {
  return strings.reduce((acc, str, i) => {
    return acc + str + (values[i] ? `<mark>${values[i]}</mark>` : '');
  }, '');
}

const result = highlight`Name: ${name}, Age: ${age}`;

// 33. Default Parameters
function greet2(name = 'Guest', greeting = 'Hello') {
  return `${greeting}, ${name}!`;
}

console.log(greet2()); // Hello, Guest!
console.log(greet2('Alice')); // Hello, Alice!

// 34. Arrow Functions
const regularFunc = function() {
  console.log(arguments); // Has arguments
  console.log(this); // Dynamic this
};

const arrowFunc = (...args) => {
  console.log(args); // Must use rest parameters
  console.log(this); // Lexical this
};

// 35. Modules (ES6)
// export const myVar = 10;
// export function myFunc() {}
// export default class MyClass {}

// import MyClass, { myVar, myFunc } from './module.js';
// import * as module from './module.js';

// Dynamic imports
// async function loadModule() {
//   const module = await import('./module.js');
//   module.myFunc();
// }

// 36. Optional Chaining
const user = {
  profile: {
    // address: { city: 'NYC' }
  }
};

console.log(user?.profile?.address?.city); // undefined (no error)
console.log(user?.getAddress?.()); // undefined (no error if method doesn't exist)

// 37. Nullish Coalescing
const value1 = null ?? 'default'; // 'default'
const value2 = 0 ?? 'default'; // 0 (0 is not nullish)
const value3 = '' ?? 'default'; // '' (empty string is not nullish)
const value4 = undefined ?? 'default'; // 'default'

// vs OR operator
const value5 = 0 || 'default'; // 'default' (0 is falsy)

// 38. BigInt
const bigNum = 9007199254740991n; // or BigInt(9007199254740991)
const anotherBig = BigInt("9007199254740991");

console.log(bigNum + 1n); // 9007199254740992n
// console.log(bigNum + 1); // TypeError: can't mix BigInt and Number

// ============================================================================
// TYPE COERCION AND COMPARISONS
// ============================================================================

// 39. Type Coercion
console.log('5' + 3); // '53' (string concatenation)
console.log('5' - 3); // 2 (numeric subtraction)
console.log('5' * '2'); // 10 (numeric multiplication)
console.log(true + 1); // 2
console.log(false + 1); // 1
console.log('5' == 5); // true (type coercion)
console.log('5' === 5); // false (no coercion)

// 40. Truthy and Falsy
// Falsy: false, 0, '', null, undefined, NaN
// Everything else is truthy

const falsyValues = [false, 0, '', null, undefined, NaN];
falsyValues.forEach(val => {
  if (!val) console.log(`${val} is falsy`);
});

// Truthy examples
console.log(!!'hello'); // true
console.log(!![]); // true
console.log(!!{}); // true

// 41. == vs ===
console.log(null == undefined); // true
console.log(null === undefined); // false

console.log(0 == false); // true
console.log(0 === false); // false

console.log('' == false); // true
console.log('' === false); // false

// 42. typeof and instanceof
console.log(typeof 42); // 'number'
console.log(typeof 'hello'); // 'string'
console.log(typeof true); // 'boolean'
console.log(typeof undefined); // 'undefined'
console.log(typeof null); // 'object' (historical bug)
console.log(typeof []); // 'object'
console.log(typeof {}); // 'object'
console.log(typeof function() {}); // 'function'

console.log([] instanceof Array); // true
console.log({} instanceof Object); // true
console.log(new Date() instanceof Date); // true

// Better array check
console.log(Array.isArray([])); // true

// 43. NaN
console.log(typeof NaN); // 'number' (quirk)
console.log(NaN === NaN); // false
console.log(Number.isNaN(NaN)); // true
console.log(Number.isNaN('hello')); // false
console.log(isNaN('hello')); // true (coerces to number first)

// 44. Primitive vs Reference
let a = 10;
let b = a;
b = 20;
console.log(a); // 10 (primitives copied by value)

let obj10 = { value: 10 };
let obj11 = obj10;
obj11.value = 20;
console.log(obj10.value); // 20 (objects copied by reference)

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

// 45. Memory Leaks - Common Causes

// Leak 1: Forgotten timers
function createLeak1() {
  const data = new Array(1000000);
  setInterval(() => {
    console.log(data[0]); // data never garbage collected
  }, 1000);
}

// Fix: Clear timer
function createNoLeak1() {
  const data = new Array(1000000);
  const timer = setInterval(() => {
    console.log(data[0]);
  }, 1000);
  
  // Clear when done
  setTimeout(() => clearInterval(timer), 5000);
}

// Leak 2: Event listeners not removed
function createLeak2() {
  const button = document.createElement('button');
  const data = new Array(1000000);
  
  button.addEventListener('click', () => {
    console.log(data[0]);
  });
  
  // Button removed but listener still holds reference
  document.body.appendChild(button);
  document.body.removeChild(button); // Memory leak!
}

// Fix: Remove event listener
function createNoLeak2() {
  const button = document.createElement('button');
  const data = new Array(1000000);
  
  const handler = () => console.log(data[0]);
  button.addEventListener('click', handler);
  
  document.body.appendChild(button);
  button.removeEventListener('click', handler);
  document.body.removeChild(button);
}

// Leak 3: Closures holding references
function createLeak3() {
  const largeData = new Array(1000000);
  
  return function() {
    // Even if largeData is not used, it's retained in closure
    console.log('hello');
  };
}

// Fix: Be explicit about what closure needs
function createNoLeak3() {
  const largeData = new Array(1000000);
  const neededValue = largeData[0];
  
  return function() {
    console.log(neededValue); // Only retains neededValue, not entire array
  };
}

// 46. WeakMap for memory optimization
class DataStore {
  constructor() {
    this.cache = new WeakMap();
  }
  
  setData(key, value) {
    this.cache.set(key, value);
  }
  
  getData(key) {
    return this.cache.get(key);
  }
}

// Objects used as keys can be garbage collected when no other references exist

// ============================================================================
// ERROR HANDLING
// ============================================================================

// 47. Try-Catch-Finally
function divide(a, b) {
  try {
    if (b === 0) throw new Error('Division by zero');
    return a / b;
  } catch (error) {
    console.error('Error:', error.message);
    return null;
  } finally {
    console.log('Cleanup always runs');
  }
}

// 48. Custom Error Types
class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ValidationError';
  }
}

class NetworkError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'NetworkError';
    this.statusCode = statusCode;
  }
}

function validateUser(user) {
  if (!user.email) {
    throw new ValidationError('Email is required');
  }
}

try {
  validateUser({});
} catch (error) {
  if (error instanceof ValidationError) {
    console.log('Validation failed:', error.message);
  } else {
    throw error; // Re-throw if not validation error
  }
}

// 49. Error Propagation in Async Code
async function fetchUser(id) {
  try {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) {
      throw new NetworkError('Failed to fetch user', response.status);
    }
    return await response.json();
  } catch (error) {
    if (error instanceof NetworkError) {
      console.error('Network error:', error.message);
    }
    throw error; // Propagate to caller
  }
}

// 50. Unhandled Promise Rejections
// Global handlers
if (typeof window !== 'undefined') {
  window.addEventListener('unhandledrejection', event => {
    console.error('Unhandled rejection:', event.reason);
    event.preventDefault();
  });
}

// ============================================================================
// PERFORMANCE OPTIMIZATION
// ============================================================================

// 51. Time Complexity Examples
// O(1) - Constant
function getFirst(arr) {
  return arr[0];
}

// O(n) - Linear
function findItem(arr, target) {
  return arr.find(item => item === target);
}

// O(n²) - Quadratic
function bubbleSort(arr) {
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
      }
    }
  }
  return arr;
}

// O(log n) - Logarithmic
function binarySearch(arr, target) {
  let left = 0, right = arr.length - 1;
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}

// 52. RequestAnimationFrame
function animateElement() {
  let position = 0;
  const element = document.getElementById('animated');
  
  function step() {
    position += 2;
    if (element) {
      element.style.transform = `translateX(${position}px)`;
    }
    
    if (position < 500) {
      requestAnimationFrame(step);
    }
  }
  
  requestAnimationFrame(step);
}

// 53. RequestIdleCallback
function performNonUrgentWork() {
  requestIdleCallback((deadline) => {
    while (deadline.timeRemaining() > 0) {
      // Do non-urgent work in small chunks
      console.log('Doing work during idle time');
    }
  });
}

// 54. Web Workers (concept)
// Main thread
// const worker = new Worker('worker.js');
// worker.postMessage({ data: [1, 2, 3, 4, 5] });
// worker.onmessage = (e) => console.log('Result:', e.data);

// worker.js file:
// self.onmessage = (e) => {
//   const result = e.data.data.reduce((a, b) => a + b, 0);
//   self.postMessage(result);
// };

// ============================================================================
// EVENT HANDLING
// ============================================================================

// 55. Event Bubbling and Capturing
const grandparent = document.createElement('div');
const parent = document.createElement('div');
const child = document.createElement('button');

grandparent.appendChild(parent);
parent.appendChild(child);

// Bubbling (default) - inner to outer
child.addEventListener('click', (e) => {
  console.log('Child clicked (bubble)');
});
parent.addEventListener('click', (e) => {
  console.log('Parent clicked (bubble)');
});
grandparent.addEventListener('click', (e) => {
  console.log('Grandparent clicked (bubble)');
});

// Capturing - outer to inner
grandparent.addEventListener('click', (e) => {
  console.log('Grandparent clicked (capture)');
}, true);
parent.addEventListener('click', (e) => {
  console.log('Parent clicked (capture)');
}, true);
child.addEventListener('click', (e) => {
  console.log('Child clicked (capture)');
}, true);

// Output when child clicked:
// Grandparent (capture) -> Parent (capture) -> Child (capture) ->
// Child (bubble) -> Parent (bubble) -> Grandparent (bubble)

// 56. Event Delegation
const list = document.createElement('ul');
list.innerHTML = `
  <li data-id="1">Item 1</li>
  <li data-id="2">Item 2</li>
  <li data-id="3">Item 3</li>
`;

// Instead of adding listener to each li
list.addEventListener('click', (e) => {
  if (e.target.tagName === 'LI') {
    console.log('Clicked item:', e.target.dataset.id);
  }
});

// 57. preventDefault and stopPropagation
const link = document.createElement('a');
link.href = 'https://example.com';

link.addEventListener('click', (e) => {
  e.preventDefault(); // Prevents navigation
  console.log('Link clicked but not navigating');
});

const outerDiv = document.createElement('div');
const innerButton = document.createElement('button');
outerDiv.appendChild(innerButton);

innerButton.addEventListener('click', (e) => {
  e.stopPropagation(); // Prevents bubbling to outerDiv
  console.log('Button clicked');
});

outerDiv.addEventListener('click', () => {
  console.log('This will not fire when button clicked');
});

// 58. Custom Events
const customEvent = new CustomEvent('userLoggedIn', {
  detail: { username: 'alice', timestamp: Date.now() },
  bubbles: true,
  cancelable: true
});

document.addEventListener('userLoggedIn', (e) => {
  console.log('User logged in:', e.detail.username);
});

// document.dispatchEvent(customEvent);

// 59. Passive Event Listeners
// Improves scroll performance
document.addEventListener('scroll', (e) => {
  console.log('Scrolling');
}, { passive: true }); // Tells browser preventDefault won't be called

// 60. Removing Event Listeners
const button = document.createElement('button');

function handleClick(e) {
  console.log('Clicked');
}

button.addEventListener('click', handleClick);

// Must use same function reference to remove
button.removeEventListener('click', handleClick);

// Anonymous functions can't be removed
button.addEventListener('click', () => console.log('Can\'t remove this'));

// ============================================================================
// DOM MANIPULATION
// ============================================================================

// 61. DOM Traversal
const element = document.createElement('div');
element.innerHTML = `
  <div class="parent">
    <div class="child1">Child 1</div>
    <div class="child2">Child 2</div>
  </div>
`;

// Navigation
// element.parentNode
// element.parentElement
// element.children
// element.childNodes (includes text nodes)
// element.firstChild
// element.firstElementChild
// element.lastChild
// element.lastElementChild
// element.nextSibling
// element.nextElementSibling
// element.previousSibling
// element.previousElementSibling

// 62. Creating and Modifying Elements
const newDiv = document.createElement('div');
newDiv.className = 'my-class';
newDiv.id = 'my-id';
newDiv.textContent = 'Hello World';
newDiv.innerHTML = '<span>Hello</span>';
newDiv.setAttribute('data-custom', 'value');

const newText = document.createTextNode('Text node');
newDiv.appendChild(newText);

// 63. DocumentFragment (efficient DOM manipulation)
function addMultipleElements() {
  const fragment = document.createDocumentFragment();
  
  // Add multiple elements to fragment (off-screen)
  for (let i = 0; i < 1000; i++) {
    const div = document.createElement('div');
    div.textContent = `Item ${i}`;
    fragment.appendChild(div);
  }
  
  // Single reflow/repaint
  // document.body.appendChild(fragment);
}

// 64. Reflow and Repaint Optimization
// Bad - causes multiple reflows
function badUpdate() {
  const el = document.getElementById('myElement');
  el.style.width = '100px';  // reflow
  el.style.height = '100px'; // reflow
  el.style.margin = '10px';  // reflow
}

// Good - single reflow
function goodUpdate() {
  const el = document.getElementById('myElement');
  el.style.cssText = 'width: 100px; height: 100px; margin: 10px;';
  // or use classes
  // el.className = 'updated-styles';
}

// Batch DOM reads and writes
function batchedOperations() {
  const elements = document.querySelectorAll('.item');
  
  // Read phase
  const heights = Array.from(elements).map(el => el.offsetHeight);
  
  // Write phase
  elements.forEach((el, i) => {
    el.style.height = heights[i] + 10 + 'px';
  });
}

// 65. IntersectionObserver
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      console.log('Element is visible:', entry.target);
      // Lazy load images, infinite scroll, etc.
    }
  });
}, {
  threshold: 0.5, // 50% visible
  rootMargin: '0px'
});

// const targetElement = document.querySelector('.observe-me');
// observer.observe(targetElement);

// 66. MutationObserver
const mutationObserver = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    console.log('DOM changed:', mutation.type);
    if (mutation.type === 'childList') {
      console.log('Children modified');
    }
    if (mutation.type === 'attributes') {
      console.log('Attributes modified');
    }
  });
});

// const observedElement = document.getElementById('observed');
// mutationObserver.observe(observedElement, {
//   attributes: true,
//   childList: true,
//   subtree: true
// });

// ============================================================================
// MODERN API KNOWLEDGE
// ============================================================================

// 67. Fetch API
async function fetchExample() {
  try {
    const response = await fetch('https://api.example.com/data', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer token'
      },
      body: JSON.stringify({ key: 'value' }),
      mode: 'cors',
      credentials: 'include'
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Fetch error:', error);
  }
}

// Abort fetch request
function abortableFetch() {
  const controller = new AbortController();
  const signal = controller.signal;
  
  fetch('https://api.example.com/data', { signal })
    .then(response => response.json())
    .catch(error => {
      if (error.name === 'AbortError') {
        console.log('Fetch aborted');
      }
    });
  
  // Abort after 5 seconds
  setTimeout(() => controller.abort(), 5000);
}

// 68. Streams API
async function streamExample() {
  const response = await fetch('https://example.com/large-file');
  const reader = response.body.getReader();
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    console.log('Received chunk:', value);
  }
}

// Create custom stream
const readableStream = new ReadableStream({
  start(controller) {
    controller.enqueue('chunk 1');
    controller.enqueue('chunk 2');
    controller.close();
  }
});

// 69. Web Storage
// localStorage - persists across sessions
localStorage.setItem('key', 'value');
const value = localStorage.getItem('key');
localStorage.removeItem('key');
localStorage.clear();

// sessionStorage - cleared when tab closes
sessionStorage.setItem('sessionKey', 'sessionValue');

// Store objects (must serialize)
const userObj = { name: 'Alice', age: 30 };
localStorage.setItem('user', JSON.stringify(userObj));
const retrievedUser = JSON.parse(localStorage.getItem('user'));

// Storage event (fired on other tabs)
window.addEventListener('storage', (e) => {
  console.log('Storage changed:', e.key, e.oldValue, e.newValue);
});

// 70. IndexedDB (basic example)
function openDatabase() {
  const request = indexedDB.open('MyDatabase', 1);
  
  request.onerror = () => console.error('Database error');
  
  request.onsuccess = (event) => {
    const db = event.target.result;
    console.log('Database opened');
  };
  
  request.onupgradeneeded = (event) => {
    const db = event.target.result;
    const objectStore = db.createObjectStore('users', { keyPath: 'id' });
    objectStore.createIndex('email', 'email', { unique: true });
  };
}

// 71. WebSockets
const socket = new WebSocket('wss://example.com/socket');

socket.addEventListener('open', () => {
  console.log('Connected');
  socket.send('Hello Server');
});

socket.addEventListener('message', (event) => {
  console.log('Message:', event.data);
});

socket.addEventListener('error', (error) => {
  console.error('WebSocket error:', error);
});

socket.addEventListener('close', () => {
  console.log('Disconnected');
});

// ============================================================================
// DESIGN PATTERNS
// ============================================================================

// 72. Module Pattern
const myModulePattern = (function() {
  let privateVar = 0;
  
  function privateMethod() {
    return privateVar++;
  }
  
  return {
    publicMethod: function() {
      return privateMethod();
    },
    getPrivateVar: function() {
      return privateVar;
    }
  };
})();

// 73. Revealing Module Pattern
const revealingModule = (function() {
  let privateCounter = 0;
  
  function increment() {
    privateCounter++;
  }
  
  function getCount() {
    return privateCounter;
  }
  
  function reset() {
    privateCounter = 0;
  }
  
  // Reveal public interface
  return {
    increment,
    getCount,
    reset
  };
})();

// 74. Singleton Pattern
const Singleton = (function() {
  let instance;
  
  function createInstance() {
    return {
      data: 'I am the instance',
      method: function() {
        return this.data;
      }
    };
  }
  
  return {
    getInstance: function() {
      if (!instance) {
        instance = createInstance();
      }
      return instance;
    }
  };
})();

const instance1 = Singleton.getInstance();
const instance2 = Singleton.getInstance();
console.log(instance1 === instance2); // true

// 75. Factory Pattern
class Car {
  constructor(options) {
    this.doors = options.doors || 4;
    this.color = options.color || 'white';
  }
}

class Truck {
  constructor(options) {
    this.doors = options.doors || 2;
    this.color = options.color || 'black';
    this.hasBed = true;
  }
}

class VehicleFactory {
  createVehicle(type, options) {
    switch(type) {
      case 'car':
        return new Car(options);
      case 'truck':
        return new Truck(options);
      default:
        throw new Error('Unknown vehicle type');
    }
  }
}

const factory = new VehicleFactory();
const myCar = factory.createVehicle('car', { doors: 2, color: 'red' });

// 76. Observer Pattern
class Subject {
  constructor() {
    this.observers = [];
  }
  
  subscribe(observer) {
    this.observers.push(observer);
  }
  
  unsubscribe(observer) {
    this.observers = this.observers.filter(obs => obs !== observer);
  }
  
  notify(data) {
    this.observers.forEach(observer => observer.update(data));
  }
}

class Observer {
  constructor(name) {
    this.name = name;
  }
  
  update(data) {
    console.log(`${this.name} received:`, data);
  }
}

const subject = new Subject();
const observer1 = new Observer('Observer 1');
const observer2 = new Observer('Observer 2');

subject.subscribe(observer1);
subject.subscribe(observer2);
subject.notify('Hello Observers!');

// 77. Pub/Sub Pattern
class PubSub {
  constructor() {
    this.events = {};
  }
  
  subscribe(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
    
    // Return unsubscribe function
    return () => {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    };
  }
  
  publish(event, data) {
    if (!this.events[event]) return;
    this.events[event].forEach(callback => callback(data));
  }
}

const pubsub = new PubSub();
const unsubscribe = pubsub.subscribe('userLogin', (user) => {
  console.log('User logged in:', user);
});

pubsub.publish('userLogin', { name: 'Alice' });
unsubscribe(); // Stop listening

// 78. Decorator Pattern
class Coffee {
  cost() {
    return 5;
  }
  
  description() {
    return 'Coffee';
  }
}

class MilkDecorator {
  constructor(coffee) {
    this.coffee = coffee;
  }
  
  cost() {
    return this.coffee.cost() + 2;
  }
  
  description() {
    return this.coffee.description() + ', Milk';
  }
}

class SugarDecorator {
  constructor(coffee) {
    this.coffee = coffee;
  }
  
  cost() {
    return this.coffee.cost() + 1;
  }
  
  description() {
    return this.coffee.description() + ', Sugar';
  }
}

let myCoffee = new Coffee();
myCoffee = new MilkDecorator(myCoffee);
myCoffee = new SugarDecorator(myCoffee);
console.log(myCoffee.description(), myCoffee.cost()); // Coffee, Milk, Sugar 8

// 79. Middleware Pattern
class Middleware {
  constructor() {
    this.middlewares = [];
  }
  
  use(fn) {
    this.middlewares.push(fn);
  }
  
  execute(context) {
    let index = 0;
    
    const next = () => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        middleware(context, next);
      }
    };
    
    next();
  }
}

const app = new Middleware();

app.use((ctx, next) => {
  console.log('Middleware 1 - before');
  next();
  console.log('Middleware 1 - after');
});

app.use((ctx, next) => {
  console.log('Middleware 2 - before');
  next();
  console.log('Middleware 2 - after');
});

app.execute({}); // Shows middleware execution order

// ============================================================================
// FUNCTIONAL PROGRAMMING
// ============================================================================

// 80. Pure Functions
// Impure
let count = 0;
function impureIncrement() {
  return ++count; // Modifies external state
}

// Pure
function pureIncrement(value) {
  return value + 1; // No side effects
}

// 81. Immutability
const originalArray = [1, 2, 3];

// Bad - mutates
originalArray.push(4);

// Good - creates new array
const newArray = [...originalArray, 4];
const newArray2 = originalArray.concat(4);

// Object immutability
const originalObj = { a: 1, b: 2 };

// Bad - mutates
originalObj.c = 3;

// Good - creates new object
const newObj = { ...originalObj, c: 3 };
const newObj2 = Object.assign({}, originalObj, { c: 3 });

// Deep immutability
function updateNested(obj, path, value) {
  const [first, ...rest] = path;
  
  if (rest.length === 0) {
    return { ...obj, [first]: value };
  }
  
  return {
    ...obj,
    [first]: updateNested(obj[first], rest, value)
  };
}

const nested = { user: { profile: { name: 'Alice' } } };
const updated = updateNested(nested, ['user', 'profile', 'name'], 'Bob');

// 82. Array Methods (map, filter, reduce)
const nums = [1, 2, 3, 4, 5];

// Map - transform
const doubled = nums.map(n => n * 2);

// Filter - select
const evens = nums.filter(n => n % 2 === 0);

// Reduce - accumulate
const sum = nums.reduce((acc, n) => acc + n, 0);

// Chaining
const result = nums
  .filter(n => n > 2)
  .map(n => n * 2)
  .reduce((acc, n) => acc + n, 0);

// Advanced reduce examples
const people = [
  { name: 'Alice', age: 30 },
  { name: 'Bob', age: 25 },
  { name: 'Charlie', age: 30 }
];

// Group by age
const grouped = people.reduce((acc, person) => {
  const age = person.age;
  if (!acc[age]) acc[age] = [];
  acc[age].push(person);
  return acc;
}, {});

// 83. Function Composition (revisited)
const addOne = x => x + 1;
const multiplyByTwo = x => x * 2;
const square = x => x * x;

// Manual composition
const compute = x => square(multiplyByTwo(addOne(x)));

// Generic compose (right to left)
const compose = (...fns) => x => 
  fns.reduceRight((acc, fn) => fn(acc), x);

// Generic pipe (left to right)
const pipe = (...fns) => x => 
  fns.reduce((acc, fn) => fn(acc), x);

const composed = compose(square, multiplyByTwo, addOne);
const piped = pipe(addOne, multiplyByTwo, square);

console.log(composed(3)); // ((3 + 1) * 2)^2 = 64
console.log(piped(3)); // ((3 + 1) * 2)^2 = 64

// 84. Point-Free Style
// Not point-free
const numbers = [1, 2, 3];
const doubled = numbers.map(n => n * 2);

// Point-free (no explicit parameters)
const double = n => n * 2;
const doubled2 = numbers.map(double);

// ============================================================================
// SECURITY
// ============================================================================

// 85. XSS Prevention
// Bad - vulnerable to XSS
function renderUserInputBad(userInput) {
  document.getElementById('output').innerHTML = userInput;
  // If userInput = '<img src=x onerror=alert("XSS")>'
}

// Good - escapes HTML
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderUserInputGood(userInput) {
  document.getElementById('output').textContent = userInput;
  // or
  // document.getElementById('output').innerHTML = escapeHtml(userInput);
}

// 86. CSRF Understanding
// CSRF token should be included in forms
function submitFormWithCsrf(data) {
  const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
  
  return fetch('/api/action', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-Token': csrfToken
    },
    body: JSON.stringify(data),
    credentials: 'same-origin'
  });
}

// 87. Content Security Policy
// Set via HTTP headers or meta tag
// <meta http-equiv="Content-Security-Policy" 
//       content="default-src 'self'; script-src 'self' https://trusted.cdn.com">

// 88. Input Sanitization
function sanitizeInput(input) {
  // Remove potentially dangerous characters
  return input
    .trim()
    .replace(/[<>\"']/g, '')
    .slice(0, 100); // Length limit
}

function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

function validateUrl(url) {
  try {
    const parsed = new URL(url);
    // Only allow http/https
    return ['http:', 'https:'].includes(parsed.protocol);
  } catch {
    return false;
  }
}

// ============================================================================
// ADVANCED TOPICS
// ============================================================================

// 89. Strict Mode
'use strict';

// Prevents accidental globals
function strictExample() {
  // 'use strict'; // Can also be function-scoped
  // x = 10; // ReferenceError in strict mode
  let x = 10; // Must use declaration
}

// Prevents duplicate parameters
// function bad(a, a, b) {} // SyntaxError in strict mode

// 90. Temporal Dead Zone
{
  // console.log(x); // ReferenceError - in TDZ
  let x = 10; // TDZ ends here
  console.log(x); // 10
}

// Function expressions in TDZ
{
  // foo(); // ReferenceError
  const foo = () => console.log('foo');
  foo(); // Works
}

// 91. Event-Driven Architecture
class EventEmitter {
  constructor() {
    this.events = {};
  }
  
  on(event, listener) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(listener);
    return this;
  }
  
  once(event, listener) {
    const onceWrapper = (...args) => {
      listener.apply(this, args);
      this.off(event, onceWrapper);
    };
    return this.on(event, onceWrapper);
  }
  
  off(event, listenerToRemove) {
    if (!this.events[event]) return this;
    
    this.events[event] = this.events[event].filter(
      listener => listener !== listenerToRemove
    );
    return this;
  }
  
  emit(event, ...args) {
    if (!this.events[event]) return false;
    
    this.events[event].forEach(listener => {
      listener.apply(this, args);
    });
    return true;
  }
}

const emitter = new EventEmitter();
emitter.on('data', (data) => console.log('Received:', data));
emitter.emit('data', { value: 42 });

// ============================================================================
// PRACTICAL CODING CHALLENGES
// ============================================================================

// 92. Deep Clone
function deepClone(obj, hash = new WeakMap()) {
  // Handle primitives and null
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  
  // Handle circular references
  if (hash.has(obj)) {
    return hash.get(obj);
  }
  
  // Handle Date
  if (obj instanceof Date) {
    return new Date(obj);
  }
  
  // Handle Array
  if (Array.isArray(obj)) {
    const clone = [];
    hash.set(obj, clone);
    obj.forEach((item, i) => {
      clone[i] = deepClone(item, hash);
    });
    return clone;
  }
  
  // Handle Object
  const clone = {};
  hash.set(obj, clone);
  Object.keys(obj).forEach(key => {
    clone[key] = deepClone(obj[key], hash);
  });
  
  return clone;
}

// Test
const original = { a: 1, b: { c: 2 }, d: [1, 2, 3] };
const cloned = deepClone(original);
cloned.b.c = 999;
console.log(original.b.c); // 2 (unchanged)

// 93. Flatten Nested Array
function flatten(arr) {
  return arr.reduce((acc, item) => {
    if (Array.isArray(item)) {
      return acc.concat(flatten(item));
    }
    return acc.concat(item);
  }, []);
}

// Or using flat
const nested = [1, [2, [3, [4]]]];
console.log(nested.flat(Infinity)); // [1, 2, 3, 4]

// 94. Implement Promise.all
function promiseAll(promises) {
  return new Promise((resolve, reject) => {
    const results = [];
    let completed = 0;
    
    if (promises.length === 0) {
      resolve(results);
      return;
    }
    
    promises.forEach((promise, index) => {
      Promise.resolve(promise)
        .then(value => {
          results[index] = value;
          completed++;
          
          if (completed === promises.length) {
            resolve(results);
          }
        })
        .catch(reject);
    });
  });
}

// 95. Implement bind
Function.prototype.myBind = function(context, ...boundArgs) {
  const fn = this;
  return function(...args) {
    return fn.apply(context, [...boundArgs, ...args]);
  };
};

function greet3(greeting, punctuation) {
  return `${greeting}, ${this.name}${punctuation}`;
}

const person2 = { name: 'Alice' };
const boundGreet = greet3.myBind(person2, 'Hello');
console.log(boundGreet('!')); // Hello, Alice!

// 96. Implement call
Function.prototype.myCall = function(context, ...args) {
  context = context || globalThis;
  const fnSymbol = Symbol();
  context[fnSymbol] = this;
  const result = context[fnSymbol](...args);
  delete context[fnSymbol];
  return result;
};

// 97. Implement apply
Function.prototype.myApply = function(context, args = []) {
  context = context || globalThis;
  const fnSymbol = Symbol();
  context[fnSymbol] = this;
  const result = context[fnSymbol](...args);
  delete context[fnSymbol];
  return result;
};

// 98. Implement Event Emitter (comprehensive)
class EventEmitter2 {
  constructor() {
    this.events = {};
  }
  
  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }
  
  off(event, callback) {
    if (!this.events[event]) return;
    this.events[event] = this.events[event].filter(cb => cb !== callback);
  }
  
  emit(event, ...args) {
    if (!this.events[event]) return;
    this.events[event].forEach(callback => callback(...args));
  }
  
  once(event, callback) {
    const wrapper = (...args) => {
      callback(...args);
      this.off(event, wrapper);
    };
    this.on(event, wrapper);
  }
}

// 99. LRU Cache Implementation
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
  }
  
  get(key) {
    if (!this.cache.has(key)) return -1;
    
    // Move to end (most recently used)
    const value = this.cache.get(key);
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }
  
  put(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.capacity) {
      // Remove least recently used (first item)
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }
}

const lru = new LRUCache(2);
lru.put(1, 1);
lru.put(2, 2);
console.log(lru.get(1)); // 1
lru.put(3, 3); // Evicts key 2
console.log(lru.get(2)); // -1 (not found)

// 100. Algorithm: Two Sum
function twoSum(nums, target) {
  const map = new Map();
  
  for (let i = 0; i < nums.length; i++) {
    const complement = target - nums[i];
    if (map.has(complement)) {
      return [map.get(complement), i];
    }
    map.set(nums[i], i);
  }
  
  return [];
}

console.log(twoSum([2, 7, 11, 15], 9)); // [0, 1]

// 101. Algorithm: Valid Parentheses
function isValidParentheses(s) {
  const stack = [];
  const pairs = { ')': '(', '}': '{', ']': '[' };
  
  for (let char of s) {
    if (char === '(' || char === '{' || char === '[') {
      stack.push(char);
    } else {
      if (stack.length === 0 || stack.pop() !== pairs[char]) {
        return false;
      }
    }
  }
  
  return stack.length === 0;
}

console.log(isValidParentheses('()[]{}')); // true
console.log(isValidParentheses('([)]')); // false

// 102. Algorithm: Merge Two Sorted Arrays
function mergeSortedArrays(arr1, arr2) {
  const result = [];
  let i = 0, j = 0;
  
  while (i < arr1.length && j < arr2.length) {
    if (arr1[i] < arr2[j]) {
      result.push(arr1[i++]);
    } else {
      result.push(arr2[j++]);
    }
  }
  
  return result.concat(arr1.slice(i)).concat(arr2.slice(j));
}

console.log(mergeSortedArrays([1, 3, 5], [2, 4, 6])); // [1, 2, 3, 4, 5, 6]

// 103. Algorithm: Find Duplicate in Array
function findDuplicate(nums) {
  const seen = new Set();
  
  for (let num of nums) {
    if (seen.has(num)) return num;
    seen.add(num);
  }
  
  return -1;
}

// Floyd's Cycle Detection (O(1) space)
function findDuplicateCycle(nums) {
  let slow = nums[0];
  let fast = nums[0];
  
  // Find intersection point
  do {
    slow = nums[slow];
    fast = nums[nums[fast]];
  } while (slow !== fast);
  
  // Find entrance to cycle
  slow = nums[0];
  while (slow !== fast) {
    slow = nums[slow];
    fast = nums[fast];
  }
  
  return slow;
}

// 104. Algorithm: Reverse Linked List
class ListNode {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

function reverseLinkedList(head) {
  let prev = null;
  let current = head;
  
  while (current !== null) {
    const next = current.next;
    current.next = prev;
    prev = current;
    current = next;
  }
  
  return prev;
}

// 105. Algorithm: Binary Search
function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;
  
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    
    if (arr[mid] === target) {
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  
  return -1;
}

// 106. Algorithm: Quick Sort
function quickSort(arr) {
  if (arr.length <= 1) return arr;
  
  const pivot = arr[arr.length - 1];
  const left = [];
  const right = [];
  
  for (let i = 0; i < arr.length - 1; i++) {
    if (arr[i] < pivot) {
      left.push(arr[i]);
    } else {
      right.push(arr[i]);
    }
  }
  
  return [...quickSort(left), pivot, ...quickSort(right)];
}

console.log(quickSort([3, 6, 8, 10, 1, 2, 1])); // [1, 1, 2, 3, 6, 8, 10]

// 107. Algorithm: Maximum Subarray (Kadane's)
function maxSubArray(nums) {
  let maxSoFar = nums[0];
  let maxEndingHere = nums[0];
  
  for (let i = 1; i < nums.length; i++) {
    maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
    maxSoFar = Math.max(maxSoFar, maxEndingHere);
  }
  
  return maxSoFar;
}

console.log(maxSubArray([-2, 1, -3, 4, -1, 2, 1, -5, 4])); // 6 ([4,-1,2,1])

// 108. String Manipulation: Anagram Check
function isAnagram(s1, s2) {
  if (s1.length !== s2.length) return false;
  
  const count = {};
  
  for (let char of s1) {
    count[char] = (count[char] || 0) + 1;
  }
  
  for (let char of s2) {
    if (!count[char]) return false;
    count[char]--;
  }
  
  return true;
}

console.log(isAnagram('listen', 'silent')); // true

// 109. String Manipulation: First Non-Repeating Character
function firstNonRepeatingChar(str) {
  const count = {};
  
  for (let char of str) {
    count[char] = (count[char] || 0) + 1;
  }
  
  for (let i = 0; i < str.length; i++) {
    if (count[str[i]] === 1) {
      return str[i];
    }
  }
  
  return null;
}

console.log(firstNonRepeatingChar('leetcode')); // 'l'

// 110. Rate Limiter Implementation
class RateLimiter {
  constructor(maxRequests, windowMs) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = [];
  }
  
  allowRequest() {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    
    // Remove old requests
    this.requests = this.requests.filter(time => time > windowStart);
    
    if (this.requests.length < this.maxRequests) {
      this.requests.push(now);
      return true;
    }
    
    return false;
  }
}

const limiter = new RateLimiter(3, 1000); // 3 requests per second
console.log(limiter.allowRequest()); // true
console.log(limiter.allowRequest()); // true
console.log(limiter.allowRequest()); // true
console.log(limiter.allowRequest()); // false

// ============================================================================
// BONUS: Common Interview Questions
// ============================================================================

// 111. Difference between var, let, and const
function varLetConstDemo() {
  // var: function-scoped, hoisted, can be redeclared
  var x = 1;
  var x = 2; // OK
  
  // let: block-scoped, not hoisted (TDZ), cannot be redeclared
  let y = 1;
  // let y = 2; // SyntaxError
  
  // const: block-scoped, not hoisted (TDZ), cannot be reassigned
  const z = 1;
  // z = 2; // TypeError
  
  // But const objects/arrays can be mutated
  const obj = { a: 1 };
  obj.a = 2; // OK
}

// 112. What is a closure? (practical example)
function createSecret(secret) {
  return {
    getSecret: () => secret,
    setSecret: (newSecret) => { secret = newSecret; }
  };
}

const mySecret = createSecret('password123');
console.log(mySecret.getSecret()); // 'password123'
mySecret.setSecret('newPassword');
console.log(mySecret.getSecret()); // 'newPassword'

// 113. Explain 'this' in different contexts
const thisExamples = {
  // Regular function
  regularMethod: function() {
    return this; // thisExamples object
  },
  
  // Arrow function
  arrowMethod: () => {
    return this; // Lexical this (global/undefined)
  },
  
  // Nested example
  nestedMethod: function() {
    const inner = () => this; // Inherits this from nestedMethod
    return inner();
  }
};

// 114. Event loop explanation with code
console.log('1');

setTimeout(() => console.log('2'), 0);

Promise.resolve().then(() => console.log('3'));

console.log('4');

// Output: 1, 4, 3, 2
// Explanation:
// 1. Synchronous code runs first: 1, 4
// 2. Microtasks (Promises) run: 3
// 3. Macrotasks (setTimeout) run: 2

// 115. Prototypal inheritance explained
function ParentConstructor(name) {
  this.name = name;
}

ParentConstructor.prototype.greet = function() {
  return `Hello, ${this.name}`;
};

function ChildConstructor(name, age) {
  ParentConstructor.call(this, name);
  this.age = age;
}

ChildConstructor.prototype = Object.create(ParentConstructor.prototype);
ChildConstructor.prototype.constructor = ChildConstructor;

ChildConstructor.prototype.introduce = function() {
  return `${this.greet()}, I'm ${this.age} years old`;
};

const child = new ChildConstructor('Alice', 10);
console.log(child.introduce()); // Hello, Alice, I'm 10 years old

// 116. Async/await error handling patterns
async function errorHandlingPatterns() {
  // Pattern 1: Try-catch
  try {
    const data = await fetchData();
    return data;
  } catch (error) {
    console.error(error);
    throw error;
  }
  
  // Pattern 2: Promise catch
  return fetchData().catch(error => {
    console.error(error);
    return null;
  });
  
  // Pattern 3: Multiple awaits with single try-catch
  try {
    const user = await fetchUser();
    const posts = await fetchPosts(user.id);
    const comments = await fetchComments(posts[0].id);
    return { user, posts, comments };
  } catch (error) {
    console.error('Error in fetch chain:', error);
    throw error;
  }
}

// 117. Implement a simple Promise
class SimplePromise {
  constructor(executor) {
    this.state = 'pending';
    this.value = undefined;
    this.handlers = [];
    
    const resolve = (value) => {
      if (this.state !== 'pending') return;
      this.state = 'fulfilled';
      this.value = value;
      this.handlers.forEach(handler => handler.onFulfilled(value));
    };
    
    const reject = (reason) => {
      if (this.state !== 'pending') return;
      this.state = 'rejected';
      this.value = reason;
      this.handlers.forEach(handler => handler.onRejected(reason));
    };
    
    try {
      executor(resolve, reject);
    } catch (error) {
      reject(error);
    }
  }
  
  then(onFulfilled, onRejected) {
    return new SimplePromise((resolve, reject) => {
      const handler = {
        onFulfilled: (value) => {
          try {
            const result = onFulfilled ? onFulfilled(value) : value;
            resolve(result);
          } catch (error) {
            reject(error);
          }
        },
        onRejected: (reason) => {
          try {
            const result = onRejected ? onRejected(reason) : reason;
            reject(result);
          } catch (error) {
            reject(error);
          }
        }
      };
      
      if (this.state === 'fulfilled') {
        handler.onFulfilled(this.value);
      } else if (this.state === 'rejected') {
        handler.onRejected(this.value);
      } else {
        this.handlers.push(handler);
      }
    });
  }
}

// ============================================================================
// END OF EXAMPLES
// ============================================================================

/* 
 * This comprehensive guide covers:
 * - Core language fundamentals (execution contexts, scope, closures, this)
 * - Asynchronous JavaScript (promises, async/await, event loop)
 * - Advanced functions (currying, composition, debounce, throttle, memoization)
 * - Objects and data structures (Maps, Sets, WeakMaps, Symbols, Proxies)
 * - ES6+ features (destructuring, spread/rest, optional chaining)
 * - Type coercion and comparisons
 * - Memory management and garbage collection
 * - Error handling patterns
 * - Performance optimization
 * - Event handling and DOM manipulation
 * - Modern APIs (Fetch, Streams, Storage, WebSockets)
 * - Design patterns (Module, Singleton, Factory, Observer, Pub/Sub)
 * - Functional programming concepts
 * - Security best practices
 * - Advanced topics and practical algorithms
 * - Common interview questions with solutions
 * 
 * Practice these examples, understand the concepts, and be ready to:
 * - Explain trade-offs between different approaches
 * - Discuss performance implications
 * - Describe real-world use cases
 * - Solve coding problems on the spot
 * - Debug and optimize code
 * 
 * Good luck with your interview!
 */