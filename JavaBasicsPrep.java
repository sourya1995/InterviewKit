// ============================================================================
// JAVA INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
// For 8+ Years SDE Experience
// ============================================================================

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.util.stream.*;
import java.util.function.*;
import java.util.Optional;
import java.lang.ref.*;
import java.lang.reflect.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.nio.channels.*;
import java.time.*;
import java.util.concurrent.CompletableFuture;

// ============================================================================
// SECTION 1: CORE JAVA FUNDAMENTALS
// ============================================================================

// ── 1.1 Primitive Types vs Object Types ──────────────────────────────────────

class PrimitivesAndBoxing {

    // Java has 8 primitives: byte, short, int, long, float, double, char, boolean
    // Autoboxing: automatic conversion between primitive and wrapper class

    public void boxingUnboxingExample() {
        int primitive = 42;
        Integer boxed = primitive;           // Autoboxing: int → Integer (heap allocation)
        int unboxed = boxed;                 // Unboxing: Integer → int

        // Integer cache: -128 to 127 are cached (same reference)
        Integer a = 127;
        Integer b = 127;
        System.out.println(a == b);          // true  (cached)

        Integer c = 128;
        Integer d = 128;
        System.out.println(c == d);          // false (different heap objects!)
        System.out.println(c.equals(d));     // true  (always use .equals() for objects)

        // NullPointerException trap with unboxing
        Integer nullableInt = null;
        // int danger = nullableInt;         // NullPointerException!
    }

    // Pass by value (Java is ALWAYS pass-by-value, even for objects — passes reference value)
    public void passByValueDemo() {
        int x = 10;
        modifyPrimitive(x);
        System.out.println(x);              // Still 10

        StringBuilder sb = new StringBuilder("hello");
        modifyObject(sb);
        System.out.println(sb);             // "hello world" (reference's object modified)

        reassignObject(sb);
        System.out.println(sb);             // "hello world" (local reassignment doesn't affect caller)
    }

    private void modifyPrimitive(int val) { val = 99; }
    private void modifyObject(StringBuilder s) { s.append(" world"); }
    private void reassignObject(StringBuilder s) { s = new StringBuilder("new"); }
}


// ── 1.2 String Internals ─────────────────────────────────────────────────────

class StringInternals {

    // String is immutable → thread-safe, cacheable hashCode
    // String Pool (intern pool): string literals are cached in heap's string pool

    public void stringPoolDemo() {
        String a = "hello";
        String b = "hello";
        String c = new String("hello");     // Forces new heap object

        System.out.println(a == b);         // true  (both from pool)
        System.out.println(a == c);         // false (c is new heap object)
        System.out.println(a.equals(c));    // true

        String interned = c.intern();       // Puts c into pool, returns pool reference
        System.out.println(a == interned);  // true
    }

    // StringBuilder vs StringBuffer vs String concatenation
    public void stringPerformance() {
        // Bad: creates N intermediate String objects in loop
        String result = "";
        for (int i = 0; i < 1000; i++) {
            result += i;                    // O(n²) — avoid!
        }

        // Good: StringBuilder (not thread-safe, use in single-thread)
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb.append(i);                   // O(n)
        }
        String good = sb.toString();

        // StringBuffer: thread-safe version of StringBuilder (synchronized methods)
        // Use only when shared between threads (rare)
        StringBuffer threadSafe = new StringBuffer();
        threadSafe.append("safe");

        // Java 8+ String.join / String.format
        String joined = String.join(", ", "a", "b", "c");  // "a, b, c"
        String formatted = String.format("Name: %s, Age: %d", "Sourya", 30);
    }

    // String methods frequently tested
    public void commonStringMethods() {
        String s = "  Hello World  ";
        s.trim();                           // "Hello World"
        s.strip();                          // Java 11+, handles Unicode whitespace
        s.toLowerCase();
        s.toUpperCase();
        s.charAt(0);
        s.substring(2, 7);
        s.contains("World");
        s.startsWith("Hello");
        s.split(" ");
        s.replace("World", "Java");
        s.indexOf("o");
        s.isEmpty();                        // length == 0
        s.isBlank();                        // Java 11+, only whitespace
        s.repeat(3);                        // Java 11+
        "abc".chars().toArray();            // IntStream of char codes
    }
}


// ── 1.3 Object Class Methods (equals, hashCode, toString, clone) ─────────────

class ObjectMethods {

    // equals() and hashCode() CONTRACT:
    //   If a.equals(b) → a.hashCode() == b.hashCode()
    //   Reverse NOT required. But violation breaks HashMap/HashSet!

    static class Point {
        int x, y;

        Point(int x, int y) { this.x = x; this.y = y; }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;              // Same reference
            if (!(o instanceof Point)) return false; // Type check
            Point p = (Point) o;
            return this.x == p.x && this.y == p.y;
        }

        @Override
        public int hashCode() {
            return Objects.hash(x, y);              // Consistent with equals
        }

        @Override
        public String toString() {
            return "Point(" + x + ", " + y + ")";
        }

        // Deep clone with Cloneable (shallow by default)
        @Override
        protected Object clone() throws CloneNotSupportedException {
            return super.clone();                   // Works since x,y are primitives
        }
    }

    public void equalsHashCodeTrap() {
        Point p1 = new Point(1, 2);
        Point p2 = new Point(1, 2);

        Set<Point> set = new HashSet<>();
        set.add(p1);
        System.out.println(set.contains(p2));      // true (because we overrode both)

        // Without overriding hashCode:
        // hashCode differs → different bucket → contains() returns false even if equals() is true!
    }
}


// ── 1.4 Immutability ─────────────────────────────────────────────────────────

// Rules for creating an immutable class:
// 1. Declare class as final
// 2. All fields private and final
// 3. No setters
// 4. Deep copy mutable fields in constructor and getters

final class ImmutableEmployee {
    private final String name;
    private final List<String> skills;       // Mutable — must defend!

    public ImmutableEmployee(String name, List<String> skills) {
        this.name = name;
        this.skills = new ArrayList<>(skills); // Defensive copy in constructor
    }

    public String getName() { return name; }

    public List<String> getSkills() {
        return Collections.unmodifiableList(skills); // Don't expose mutable reference
    }
}


// ============================================================================
// SECTION 2: OOP CONCEPTS (DEEP DIVE)
// ============================================================================

// ── 2.1 Abstract Class vs Interface ──────────────────────────────────────────

abstract class AbstractAnimal {
    private String name;

    AbstractAnimal(String name) { this.name = name; }

    // Concrete method — shared logic
    public String getName() { return name; }

    // Abstract method — subclasses MUST implement
    public abstract String makeSound();

    // Template Method pattern
    public final void performAction() {
        System.out.println(getName() + " says: " + makeSound());
    }
}

interface Flyable {
    // Java 8+: default methods (add behavior without breaking implementations)
    default String fly() { return "flying"; }

    // Java 8+: static methods in interface
    static String description() { return "Something that can fly"; }

    // Java 9+: private methods (helper for default methods)
    // private void helper() {}
}

interface Swimmable {
    default String swim() { return "swimming"; }
}

// Multiple interface implementation (Java's answer to multiple inheritance)
class Duck extends AbstractAnimal implements Flyable, Swimmable {
    Duck() { super("Duck"); }

    @Override
    public String makeSound() { return "Quack"; }
}

// Key differences:
// Abstract class: single inheritance, can have state (fields), constructors, access modifiers
// Interface: multiple implementation, no state (pre-Java 8), all public by default


// ── 2.2 Polymorphism & Method Dispatch ───────────────────────────────────────

class PolymorphismDemo {

    static class Shape {
        public double area() { return 0; }           // overrideable
        public static String type() { return "Shape"; } // NOT overrideable (hidden, not overridden)
    }

    static class Circle extends Shape {
        double radius;
        Circle(double r) { this.radius = r; }

        @Override
        public double area() { return Math.PI * radius * radius; }

        // Static method hiding — NOT polymorphism!
        public static String type() { return "Circle"; }
    }

    public void dynamicDispatch() {
        Shape s = new Circle(5);
        System.out.println(s.area());        // Circle's area() — runtime dispatch (polymorphism)
        System.out.println(Shape.type());    // "Shape" — compile-time (static binding)
        System.out.println(Circle.type());   // "Circle"
        // s.type() would call Shape.type() since reference type is Shape
    }

    // Covariant return types (Java 5+)
    static class Base {
        public Base create() { return new Base(); }
    }
    static class Derived extends Base {
        @Override
        public Derived create() { return new Derived(); } // Covariant — more specific return type OK
    }
}


// ── 2.3 Composition vs Inheritance ───────────────────────────────────────────

// Prefer composition over inheritance (Effective Java Item 18)
// Inheritance: "is-a" relationship, tight coupling, fragile base class problem
// Composition: "has-a" relationship, loose coupling, flexible

class Engine {
    public void start() { System.out.println("Engine started"); }
    public void stop()  { System.out.println("Engine stopped"); }
}

class Car {   // Composition
    private final Engine engine = new Engine();  // Has-a Engine
    private String model;

    Car(String model) { this.model = model; }

    public void start() { engine.start(); }  // Delegates to component
    public void stop()  { engine.stop(); }
}


// ── 2.4 SOLID Principles ─────────────────────────────────────────────────────

// S — Single Responsibility
class UserService {    // Only handles user business logic
    void createUser(String name) { /* ... */ }
}
class UserRepository { // Only handles persistence
    void save(String user) { /* ... */ }
}

// O — Open/Closed Principle
interface Discount {
    double apply(double price);
}
class SeasonalDiscount implements Discount {
    public double apply(double price) { return price * 0.9; }
}
class LoyaltyDiscount implements Discount {
    public double apply(double price) { return price * 0.85; }
}
// Add new discount types without modifying existing code

// L — Liskov Substitution
// Derived class must be substitutable for base class
class Rectangle2 {
    int width, height;
    void setWidth(int w)  { this.width = w; }
    void setHeight(int h) { this.height = h; }
    int area() { return width * height; }
}
// Square violates LSP if it overrides setters — use composition instead

// I — Interface Segregation
interface Printable   { void print(); }
interface Saveable    { void save(); }
interface Searchable  { void search(); }
// Don't force implementing classes to depend on methods they don't use

// D — Dependency Inversion
interface OrderRepository {
    void saveOrder(String order);
}
class OrderService {
    private final OrderRepository repo; // Depends on abstraction, not concrete class
    OrderService(OrderRepository repo) { this.repo = repo; }
    void placeOrder(String order) { repo.saveOrder(order); }
}


// ============================================================================
// SECTION 3: GENERICS
// ============================================================================

class GenericsDeep {

    // Generic class
    static class Pair<A, B> {
        private final A first;
        private final B second;

        Pair(A first, B second) {
            this.first = first;
            this.second = second;
        }

        public A getFirst() { return first; }
        public B getSecond() { return second; }
    }

    // Generic method
    public static <T extends Comparable<T>> T max(T a, T b) {
        return a.compareTo(b) >= 0 ? a : b;
    }

    // Bounded wildcards — PECS: Producer Extends, Consumer Super

    // Upper bounded wildcard (? extends T) — read from (covariance)
    public double sumList(List<? extends Number> list) {
        double sum = 0;
        for (Number n : list) sum += n.doubleValue();
        return sum;
        // list.add(1.0);   // COMPILE ERROR — can't add to ? extends
    }

    // Lower bounded wildcard (? super T) — write to (contravariance)
    public void addNumbers(List<? super Integer> list) {
        list.add(1);
        list.add(2);
        // Integer n = list.get(0);  // COMPILE ERROR — can't read specific type
    }

    // Unbounded wildcard (?) — read as Object
    public void printList(List<?> list) {
        for (Object o : list) System.out.println(o);
    }

    // Type erasure: generics are compile-time only; at runtime List<String> == List<Integer>
    // This is why you can't do: new T(), instanceof List<String>, T.class

    // Reifiable types: arrays are reifiable (type info at runtime), generics are not
    // String[] arr = new String[10];      // OK
    // List<String>[] arr2 = new List[10]; // Unchecked warning

    // Generic interface with bounded type parameter
    interface Comparable2<T> {
        int compareTo(T other);
    }

    // Recursive type bound
    static <T extends Comparable<T>> T findMax(List<T> list) {
        return list.stream().max(Comparator.naturalOrder()).orElseThrow();
    }
}


// ============================================================================
// SECTION 4: COLLECTIONS FRAMEWORK
// ============================================================================

class CollectionsDeep {

    // ── List Implementations ──────────────────────────────────────────────────

    public void listComparison() {
        // ArrayList: dynamic array, O(1) random access, O(n) insert/delete mid
        List<Integer> arrayList = new ArrayList<>();
        arrayList.add(1); arrayList.add(2); arrayList.add(3);
        arrayList.get(0);  // O(1)

        // LinkedList: doubly-linked, O(1) insert/delete at head/tail, O(n) random access
        List<Integer> linkedList = new LinkedList<>();
        ((LinkedList<Integer>) linkedList).addFirst(0);   // O(1)
        ((LinkedList<Integer>) linkedList).addLast(4);    // O(1)

        // When to use LinkedList: frequent insertions/deletions at start/end
        // When to use ArrayList: frequent reads, index-based access
    }

    // ── Map Implementations ───────────────────────────────────────────────────

    public void mapComparison() {
        // HashMap: O(1) avg get/put, unordered, allows null key/value, not thread-safe
        Map<String, Integer> hashMap = new HashMap<>();
        hashMap.put("a", 1);

        // LinkedHashMap: preserves insertion order (or access order with flag)
        Map<String, Integer> linked = new LinkedHashMap<>();

        // TreeMap: sorted by key (natural order or Comparator), O(log n) ops, no null keys
        Map<String, Integer> treeMap = new TreeMap<>();

        // ConcurrentHashMap: thread-safe HashMap, no lock on read, segment locks on write
        Map<String, Integer> concurrent = new ConcurrentHashMap<>();
        concurrent.put("key", 1);
        concurrent.putIfAbsent("key", 2);           // Atomic
        concurrent.computeIfAbsent("key2", k -> k.length()); // Atomic compute

        // Hashtable: legacy, fully synchronized (prefer ConcurrentHashMap)

        // HashMap internals: array of buckets (LinkedList → TreeNode when bucket.size >= 8)
        // Load factor 0.75, resize when capacity * loadFactor exceeded
        // Java 8+: treeify bucket when 8+ entries, untreeify at 6
    }

    // ── Set Implementations ───────────────────────────────────────────────────

    public void setComparison() {
        // HashSet: backed by HashMap, O(1) add/contains/remove, unordered
        Set<String> hashSet = new HashSet<>();

        // LinkedHashSet: insertion-ordered HashSet
        Set<String> linkedSet = new LinkedHashSet<>();

        // TreeSet: sorted, backed by TreeMap, O(log n), no null
        Set<String> treeSet = new TreeSet<>();

        // EnumSet: extremely fast, bit-vector backed, for enum values
        // EnumSet<Day> weekdays = EnumSet.of(Day.MON, Day.TUE);
    }

    // ── Queue & Deque ─────────────────────────────────────────────────────────

    public void queueComparison() {
        // ArrayDeque: resizable array, O(1) add/remove at both ends, faster than LinkedList
        Deque<Integer> deque = new ArrayDeque<>();
        deque.offerFirst(1);    // Add to front
        deque.offerLast(2);     // Add to back
        deque.pollFirst();      // Remove from front
        deque.peekLast();       // Peek back

        // PriorityQueue: min-heap by default, O(log n) insert, O(1) peek, O(log n) poll
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());

        // BlockingQueue (for producer-consumer)
        BlockingQueue<String> blocking = new LinkedBlockingQueue<>(100);
    }

    // ── Utility Methods ───────────────────────────────────────────────────────

    public void collectionsUtils() {
        List<Integer> list = new ArrayList<>(Arrays.asList(3, 1, 4, 1, 5, 9));

        Collections.sort(list);                              // In-place sort
        Collections.sort(list, Comparator.reverseOrder());   // Reverse
        Collections.shuffle(list);
        Collections.reverse(list);
        Collections.min(list);
        Collections.max(list);
        Collections.frequency(list, 1);                      // Count occurrences
        Collections.unmodifiableList(list);                  // Immutable view
        Collections.synchronizedList(list);                  // Thread-safe view (coarse-grained lock)

        // Immutable collections (Java 9+)
        List<String> immutable = List.of("a", "b", "c");    // Throws on modify
        Map<String, Integer> immutableMap = Map.of("a", 1, "b", 2);
        Set<String> immutableSet = Set.of("x", "y");

        // Arrays utility
        int[] arr = {3, 1, 4};
        Arrays.sort(arr);
        Arrays.binarySearch(arr, 3);
        Arrays.fill(arr, 0);
        Arrays.copyOfRange(arr, 0, 2);
        Arrays.stream(arr).sum();
    }
}


// ============================================================================
// SECTION 5: JAVA 8+ FUNCTIONAL PROGRAMMING & STREAMS
// ============================================================================

class FunctionalAndStreams {

    // ── Functional Interfaces ─────────────────────────────────────────────────

    // java.util.function package — the core functional interfaces:
    // Function<T,R>      — takes T, returns R            → apply()
    // Predicate<T>       — takes T, returns boolean       → test()
    // Consumer<T>        — takes T, returns void          → accept()
    // Supplier<T>        — takes nothing, returns T       → get()
    // BiFunction<T,U,R>  — takes T & U, returns R
    // UnaryOperator<T>   — Function<T,T>
    // BinaryOperator<T>  — BiFunction<T,T,T>

    public void functionalInterfacesDemo() {
        Function<String, Integer> length = String::length;
        Function<Integer, Integer> doubler = x -> x * 2;
        Function<String, Integer> composed = length.andThen(doubler); // length then double
        System.out.println(composed.apply("hello")); // 10

        Predicate<String> longStr = s -> s.length() > 5;
        Predicate<String> startsA = s -> s.startsWith("A");
        Predicate<String> both = longStr.and(startsA);
        Predicate<String> either = longStr.or(startsA);
        Predicate<String> notLong = longStr.negate();

        Consumer<String> print = System.out::println;
        Consumer<String> printUpper = s -> System.out.println(s.toUpperCase());
        Consumer<String> bothActions = print.andThen(printUpper);

        Supplier<List<String>> listFactory = ArrayList::new;
        List<String> newList = listFactory.get();
    }

    // ── Lambda Expressions ────────────────────────────────────────────────────

    public void lambdaDemo() {
        // Lambdas capture effectively final variables from enclosing scope
        String prefix = "Hello";  // effectively final
        Consumer<String> greeter = name -> System.out.println(prefix + " " + name);
        // prefix = "Hi";  // COMPILE ERROR — would make prefix not effectively final

        // Method references — 4 types:
        // Static method reference
        Function<String, Integer> parseInt = Integer::parseInt;

        // Instance method reference (on specific instance)
        String str = "hello";
        Supplier<String> upper = str::toUpperCase;

        // Instance method reference (on arbitrary instance of type)
        Function<String, String> toUpper = String::toUpperCase;

        // Constructor reference
        Supplier<ArrayList<String>> listSupplier = ArrayList::new;
        Function<Integer, int[]> arrFactory = int[]::new;
    }

    // ── Optional ─────────────────────────────────────────────────────────────

    public void optionalDemo() {
        Optional<String> opt1 = Optional.of("value");
        Optional<String> opt2 = Optional.empty();
        Optional<String> opt3 = Optional.ofNullable(null);  // Empty if null

        // BAD: defeats the purpose
        if (opt1.isPresent()) {
            System.out.println(opt1.get());
        }

        // GOOD: use functional style
        opt1.ifPresent(System.out::println);
        String result = opt2.orElse("default");
        String computed = opt2.orElseGet(() -> computeDefault());
        String mapped = opt1.map(String::toUpperCase).orElse("none");
        Optional<String> filtered = opt1.filter(s -> s.length() > 3);

        // FlatMap — when mapper returns Optional<Optional<T>>
        Optional<String> chained = opt1.flatMap(s -> Optional.of(s + "!"));

        // Java 9+
        opt2.ifPresentOrElse(System.out::println, () -> System.out.println("empty"));
        opt2.or(() -> Optional.of("fallback"));     // Returns Optional

        // Don't use Optional as method parameter or field — design smell
        // Use Optional only as return type to signal possible absence
    }

    private String computeDefault() { return "computed"; }

    // ── Streams ───────────────────────────────────────────────────────────────

    public void streamsDemo() {
        List<String> names = List.of("Alice", "Bob", "Charlie", "Dave", "Eve");

        // Intermediate operations (lazy): filter, map, flatMap, distinct, sorted, peek, limit, skip
        // Terminal operations (eager, triggers pipeline): forEach, collect, reduce, count, findFirst,
        //   findAny, anyMatch, allMatch, noneMatch, min, max, toArray

        List<String> result = names.stream()
            .filter(s -> s.length() > 3)            // Keep names longer than 3 chars
            .map(String::toUpperCase)               // Transform to uppercase
            .sorted()                               // Natural order
            .collect(Collectors.toList());

        // FlatMap — flatten nested collections
        List<List<Integer>> nested = List.of(List.of(1, 2), List.of(3, 4));
        List<Integer> flat = nested.stream()
            .flatMap(Collection::stream)
            .collect(Collectors.toList());           // [1, 2, 3, 4]

        // Collectors
        Map<Integer, List<String>> byLength = names.stream()
            .collect(Collectors.groupingBy(String::length));

        Map<Boolean, List<String>> partitioned = names.stream()
            .collect(Collectors.partitioningBy(s -> s.length() > 3));

        String joined = names.stream()
            .collect(Collectors.joining(", ", "[", "]"));

        long count = names.stream().filter(s -> s.startsWith("A")).count();

        Optional<String> first = names.stream().filter(s -> s.length() > 4).findFirst();

        // Reduce
        int sum = IntStream.rangeClosed(1, 10).reduce(0, Integer::sum); // 55

        // Statistics
        IntSummaryStatistics stats = names.stream()
            .mapToInt(String::length)
            .summaryStatistics();
        stats.getAverage(); stats.getMax(); stats.getMin(); stats.getSum();

        // Parallel streams — split work across ForkJoinPool.commonPool()
        // Use for CPU-bound, stateless, large data sets; avoid for small sets or IO-bound
        long parallelCount = names.parallelStream()
            .filter(s -> s.length() > 3)
            .count();

        // Infinite streams
        Stream.iterate(0, n -> n + 1).limit(10).forEach(System.out::println);
        Stream.generate(Math::random).limit(5).forEach(System.out::println);
    }

    // ── Comparator & Comparable ───────────────────────────────────────────────

    static class Employee implements Comparable<Employee> {
        String name;
        int age;
        double salary;

        Employee(String name, int age, double salary) {
            this.name = name;
            this.age = age;
            this.salary = salary;
        }

        @Override
        public int compareTo(Employee other) {
            return this.name.compareTo(other.name); // Natural order by name
        }
    }

    public void comparatorDemo() {
        List<Employee> employees = new ArrayList<>();
        employees.add(new Employee("Zoe", 30, 90000));
        employees.add(new Employee("Alice", 25, 75000));
        employees.add(new Employee("Bob", 35, 85000));

        // Sort by salary, then by name
        employees.sort(Comparator.comparingDouble((Employee e) -> e.salary)
            .thenComparing(e -> e.name));

        // Reverse
        employees.sort(Comparator.comparingInt((Employee e) -> e.age).reversed());

        // Natural order (uses Comparable)
        Collections.sort(employees);

        // Custom Comparator with null safety
        employees.sort(Comparator.nullsFirst(Comparator.comparing(e -> e.name)));
    }
}


// ============================================================================
// SECTION 6: EXCEPTION HANDLING
// ============================================================================

class ExceptionHandling {

    // Checked exceptions: must declare or handle (IOException, SQLException)
    // Unchecked exceptions: RuntimeException subclasses (NullPointerException, etc.)
    // Error: JVM-level, don't catch (OutOfMemoryError, StackOverflowError)

    // Custom exceptions
    static class InsufficientFundsException extends Exception {
        private final double amount;

        public InsufficientFundsException(double amount) {
            super("Insufficient funds: need " + amount + " more");
            this.amount = amount;
        }

        public double getAmount() { return amount; }
    }

    static class BusinessException extends RuntimeException {
        private final String errorCode;

        public BusinessException(String errorCode, String message) {
            super(message);
            this.errorCode = errorCode;
        }

        public BusinessException(String errorCode, String message, Throwable cause) {
            super(message, cause); // Always chain the original exception!
            this.errorCode = errorCode;
        }

        public String getErrorCode() { return errorCode; }
    }

    // try-with-resources (AutoCloseable) — Java 7+
    public void tryWithResources(String path) {
        try (InputStream is = new FileInputStream(path);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {

            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch (FileNotFoundException e) {
            throw new BusinessException("FILE_NOT_FOUND", "File not found: " + path, e);
        } catch (IOException e) {
            throw new BusinessException("IO_ERROR", "IO error reading file", e);
        }
        // Both resources closed automatically, even if exception occurs
    }

    // Multi-catch (Java 7+)
    public void multiCatch() {
        try {
            // risky operations
        } catch (IllegalArgumentException | IllegalStateException e) {
            System.out.println("Either error: " + e.getMessage());
        }
    }

    // Exception chaining (always do this in wrapping scenarios)
    public void exceptionChaining(String file) throws BusinessException {
        try {
            Files.readString(Path.of(file));
        } catch (IOException e) {
            // GOOD: preserve original exception as cause
            throw new BusinessException("READ_FAIL", "Cannot read file", e);
            // BAD: throw new BusinessException("READ_FAIL", "Cannot read file");
            // — loses the original stack trace!
        }
    }

    // Finally: runs even if exception thrown; don't use for return statements
    public int finallyDemo() {
        try {
            return 1;
        } finally {
            System.out.println("Always runs");
            // return 2; // BAD: swallows exception and overrides return value!
        }
    }
}


// ============================================================================
// SECTION 7: CONCURRENCY & MULTITHREADING
// ============================================================================

class ConcurrencyDeep {

    // ── Thread Creation ───────────────────────────────────────────────────────

    public void threadCreationWays() throws InterruptedException {
        // Way 1: Extend Thread
        Thread t1 = new Thread() {
            @Override public void run() { System.out.println("Thread 1"); }
        };

        // Way 2: Implement Runnable (preferred — keeps class free for other inheritance)
        Thread t2 = new Thread(() -> System.out.println("Thread 2"));

        // Way 3: ExecutorService (preferred in production)
        ExecutorService executor = Executors.newFixedThreadPool(4);
        executor.submit(() -> System.out.println("Thread 3"));
        executor.shutdown();

        t1.start();
        t2.start();
        t1.join();  // Wait for t1 to finish
        t2.join();
    }

    // ── Thread Lifecycle ──────────────────────────────────────────────────────
    // NEW → RUNNABLE → RUNNING → BLOCKED/WAITING/TIMED_WAITING → TERMINATED

    // ── Synchronized & Locks ──────────────────────────────────────────────────

    static class Counter {
        private int count = 0;

        // Synchronized method — locks on 'this'
        public synchronized void increment() { count++; }
        public synchronized int getCount()   { return count; }

        // Synchronized block — more granular, preferred
        private final Object lock = new Object();
        public void incrementBlock() {
            synchronized (lock) {
                count++;
            }
        }
    }

    // ReentrantLock — more flexible than synchronized
    static class BetterCounter {
        private int count = 0;
        private final ReentrantLock lock = new ReentrantLock();

        public void increment() {
            lock.lock();
            try {
                count++;
            } finally {
                lock.unlock(); // ALWAYS unlock in finally!
            }
        }

        public boolean tryIncrement() {
            if (lock.tryLock()) { // Non-blocking attempt
                try {
                    count++;
                    return true;
                } finally {
                    lock.unlock();
                }
            }
            return false;
        }
    }

    // ReadWriteLock — multiple readers OR one writer
    static class Cache {
        private final Map<String, String> data = new HashMap<>();
        private final ReadWriteLock rwLock = new ReentrantReadWriteLock();

        public String get(String key) {
            rwLock.readLock().lock();      // Many threads can read simultaneously
            try {
                return data.get(key);
            } finally {
                rwLock.readLock().unlock();
            }
        }

        public void put(String key, String value) {
            rwLock.writeLock().lock();     // Only one writer at a time
            try {
                data.put(key, value);
            } finally {
                rwLock.writeLock().unlock();
            }
        }
    }

    // ── Atomic Operations ─────────────────────────────────────────────────────

    static class AtomicCounter {
        private final AtomicInteger count = new AtomicInteger(0);
        private final AtomicLong longCount = new AtomicLong(0);
        private final AtomicReference<String> ref = new AtomicReference<>("");

        public int increment()           { return count.incrementAndGet(); }
        public int addAndGet(int delta)  { return count.addAndGet(delta); }

        // Compare-and-swap (CAS) — lock-free, optimistic concurrency
        public boolean setIfZero() {
            return count.compareAndSet(0, 1); // Atomic: if current==0, set to 1
        }
    }

    // ── volatile keyword ──────────────────────────────────────────────────────

    static class VisibilityExample {
        private volatile boolean running = true;  // Visible to all threads immediately

        // Without volatile: JIT may cache 'running' in CPU register,
        // another thread's write won't be seen → infinite loop!
        public void run() {
            while (running) { /* work */ }
        }

        public void stop() { running = false; }  // Visible immediately after volatile write
    }

    // ── wait/notify (Object Monitor) ──────────────────────────────────────────

    static class ProducerConsumerMonitor {
        private final Queue<Integer> queue = new LinkedList<>();
        private final int capacity = 10;

        public synchronized void produce(int item) throws InterruptedException {
            while (queue.size() == capacity) {
                wait();                  // Release lock, wait for notify
            }
            queue.add(item);
            notifyAll();                 // Wake all waiting threads
        }

        public synchronized int consume() throws InterruptedException {
            while (queue.isEmpty()) {
                wait();
            }
            int item = queue.poll();
            notifyAll();
            return item;
        }
        // Always use while() not if() with wait() — guard against spurious wakeups!
    }

    // ── ExecutorService & Thread Pools ───────────────────────────────────────

    public void executorServiceDemo() throws Exception {
        // Fixed thread pool
        ExecutorService fixed = Executors.newFixedThreadPool(4);

        // Cached thread pool — creates threads on demand, reuses idle ones (dangerous for uncontrolled load)
        ExecutorService cached = Executors.newCachedThreadPool();

        // Single thread executor — sequential, preserves order
        ExecutorService single = Executors.newSingleThreadExecutor();

        // Scheduled executor
        ScheduledExecutorService scheduled = Executors.newScheduledThreadPool(2);
        scheduled.scheduleAtFixedRate(() -> System.out.println("tick"), 0, 1, TimeUnit.SECONDS);

        // Submit Callable (returns Future<T>)
        Future<Integer> future = fixed.submit(() -> {
            Thread.sleep(100);
            return 42;
        });
        Integer result = future.get(500, TimeUnit.MILLISECONDS); // Blocking get with timeout

        // Invoke all / any
        List<Callable<Integer>> tasks = List.of(() -> 1, () -> 2, () -> 3);
        List<Future<Integer>> results = fixed.invokeAll(tasks);
        Integer first = fixed.invokeAny(tasks); // Returns first successful result

        // Proper shutdown
        fixed.shutdown();                        // Stop accepting new tasks
        if (!fixed.awaitTermination(30, TimeUnit.SECONDS)) {
            fixed.shutdownNow();                 // Cancel running tasks
        }
    }

    // ── CompletableFuture ────────────────────────────────────────────────────

    public void completableFutureDemo() throws Exception {
        // Non-blocking async pipeline
        CompletableFuture<String> cf = CompletableFuture
            .supplyAsync(() -> "hello")                     // Async supplier
            .thenApply(String::toUpperCase)                 // Transform result
            .thenApply(s -> s + "!")
            .thenCombine(
                CompletableFuture.supplyAsync(() -> " World"),
                (a, b) -> a + b                             // Combine two futures
            )
            .exceptionally(ex -> "Error: " + ex.getMessage()); // Handle exception

        cf.thenAccept(System.out::println);                 // Consume result
        String result = cf.get();

        // Run all, wait for all
        CompletableFuture<Void> all = CompletableFuture.allOf(
            CompletableFuture.supplyAsync(() -> 1),
            CompletableFuture.supplyAsync(() -> 2)
        );

        // Return first to complete
        CompletableFuture<Object> any = CompletableFuture.anyOf(
            CompletableFuture.supplyAsync(() -> "a"),
            CompletableFuture.supplyAsync(() -> "b")
        );

        // Chain async steps
        CompletableFuture<String> chained = CompletableFuture
            .supplyAsync(() -> "user-id-123")
            .thenComposeAsync(id -> CompletableFuture.supplyAsync(() -> "User: " + id)); // flatMap equivalent

        // Handle both result and exception
        CompletableFuture<String> handled = CompletableFuture
            .supplyAsync(() -> { throw new RuntimeException("fail"); })
            .handle((result2, ex) -> ex != null ? "recovered" : result2.toString());
    }

    // ── Concurrent Collections ───────────────────────────────────────────────

    public void concurrentCollectionsDemo() {
        // ConcurrentHashMap — thread-safe, no null keys/values
        ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
        map.put("a", 1);
        map.computeIfAbsent("b", k -> 2);          // Atomic compute
        map.merge("a", 1, Integer::sum);            // Atomic merge

        // CopyOnWriteArrayList — thread-safe reads, writes create copy
        // Best for read-heavy, write-rare scenarios
        List<String> cowList = new CopyOnWriteArrayList<>();
        cowList.add("item");                       // Creates new internal array

        // BlockingQueue — producer-consumer
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(100);
        try {
            queue.put("item");                     // Blocks if full
            String item = queue.take();            // Blocks if empty
            queue.offer("item", 100, TimeUnit.MILLISECONDS); // Timed offer
            queue.poll(100, TimeUnit.MILLISECONDS); // Timed poll
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();    // Restore interrupted status!
        }

        // Semaphore — limit concurrent access
        Semaphore semaphore = new Semaphore(3);    // 3 permits
        try {
            semaphore.acquire();                   // Get a permit (blocks if none)
            // critical section — at most 3 threads here
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            semaphore.release();                   // Return permit
        }

        // CountDownLatch — wait for N events
        CountDownLatch latch = new CountDownLatch(3);
        // Worker threads call latch.countDown()
        try {
            latch.await(5, TimeUnit.SECONDS);      // Wait for all 3 countdowns
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // CyclicBarrier — all threads wait at barrier, then proceed together (reusable)
        CyclicBarrier barrier = new CyclicBarrier(4, () -> System.out.println("All ready!"));
    }

    // ── ThreadLocal ──────────────────────────────────────────────────────────

    static class ThreadLocalDemo {
        // Each thread gets its own copy — no synchronization needed
        private static final ThreadLocal<SimpleDateFormat> dateFormat =
            ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyy-MM-dd"));

        public String formatDate(Date date) {
            return dateFormat.get().format(date);  // Each thread has its own DateFormat
        }

        // IMPORTANT: In thread pool scenarios, always remove after use to prevent leaks!
        public void cleanup() {
            dateFormat.remove();
        }
    }
}


// ============================================================================
// SECTION 8: JVM INTERNALS & MEMORY MODEL
// ============================================================================

class JVMInternals {

    /*
     * JVM Memory Areas:
     * ┌─────────────────────────────────────────┐
     * │  Method Area (Metaspace in Java 8+)     │ — class metadata, static vars, constants
     * │  Heap                                   │ — all objects, GC-managed
     * │  ├── Young Generation                   │
     * │  │   ├── Eden                           │ — new objects allocated here
     * │  │   ├── Survivor S0                    │ — survived 1st GC
     * │  │   └── Survivor S1                    │
     * │  └── Old Generation (Tenured)           │ — long-lived objects
     * │  Stack (per thread)                     │ — stack frames (local vars, method calls)
     * │  PC Register (per thread)               │ — current bytecode instruction
     * │  Native Method Stack (per thread)       │ — JNI calls
     * └─────────────────────────────────────────┘
     *
     * GC Algorithms:
     * - Serial GC: single-threaded, for small apps
     * - Parallel GC: multi-threaded, throughput focused (Java 8 default)
     * - G1 GC: region-based, predictable pause times (Java 9+ default)
     * - ZGC/Shenandoah: low-latency, pauses < 10ms (Java 15+)
     *
     * GC Phases:
     * - Minor GC: collects Young Generation (fast, frequent)
     * - Major GC: collects Old Generation (slow, infrequent)
     * - Full GC: collects entire heap (stop-the-world, avoid!)
     *
     * Java Memory Model (JMM):
     * - Defines visibility rules between threads
     * - happens-before relationship:
     *   - Thread start → all actions in thread
     *   - Monitor unlock → subsequent lock
     *   - volatile write → subsequent read
     *   - Thread join → all actions in joined thread
     *
     * Class Loading:
     * Bootstrap ClassLoader → Extension/Platform ClassLoader → Application ClassLoader
     * Parent delegation model: child asks parent first
     */

    public void memoryModelDemo() {
        // Stack allocation: primitive local variables, reference variables (not the objects)
        int x = 10;           // Stack
        String s = "hello";   // Reference on stack, object in heap (string pool)

        // Heap allocation
        Object obj = new Object();  // Object in heap, reference on stack

        // Escape analysis (JIT optimization):
        // If JVM determines object doesn't escape the method, may allocate on stack
        // This is why small objects created inside methods may have no GC overhead
    }

    // Strong, Soft, Weak, Phantom references
    public void referenceTypesDemo() {
        Object obj = new Object();

        // Strong reference — prevents GC
        Object strong = obj;

        // Soft reference — GC'd only when memory is low (good for caches)
        SoftReference<Object> soft = new SoftReference<>(obj);
        Object maybeLive = soft.get();  // May return null after GC

        // Weak reference — GC'd at next GC cycle
        WeakReference<Object> weak = new WeakReference<>(obj);
        obj = null;                     // Remove strong reference
        // GC can now collect the object; weak.get() may return null

        // Phantom reference — for post-GC cleanup (finalization replacement)
        ReferenceQueue<Object> queue = new ReferenceQueue<>();
        PhantomReference<Object> phantom = new PhantomReference<>(obj, queue);
        // phantom.get() always returns null; enqueued after object collected
    }
}


// ============================================================================
// SECTION 9: DESIGN PATTERNS (JAVA IMPLEMENTATIONS)
// ============================================================================

class DesignPatterns {

    // ── Singleton (Thread-safe variants) ─────────────────────────────────────

    // Double-checked locking (DCL) — most common
    static class Singleton {
        private static volatile Singleton instance;  // volatile required!

        private Singleton() {}

        public static Singleton getInstance() {
            if (instance == null) {                  // First check (no lock)
                synchronized (Singleton.class) {
                    if (instance == null) {          // Second check (with lock)
                        instance = new Singleton();
                    }
                }
            }
            return instance;
        }
    }

    // Initialization-on-demand holder (lazy, thread-safe, elegant)
    static class SingletonHolder {
        private SingletonHolder() {}

        private static class Holder {
            static final SingletonHolder INSTANCE = new SingletonHolder();
        }

        public static SingletonHolder getInstance() { return Holder.INSTANCE; }
    }

    // Enum Singleton — Josh Bloch's recommendation (handles serialization too)
    enum EnumSingleton {
        INSTANCE;
        public void doWork() { /* ... */ }
    }

    // ── Builder Pattern ───────────────────────────────────────────────────────

    static class Person {
        private final String name;
        private final int age;
        private final String email;
        private final String phone;

        private Person(Builder b) {
            this.name  = b.name;
            this.age   = b.age;
            this.email = b.email;
            this.phone = b.phone;
        }

        static class Builder {
            private final String name;  // Required
            private int age;
            private String email;
            private String phone;

            Builder(String name) { this.name = name; }

            Builder age(int age)       { this.age = age;     return this; }
            Builder email(String e)    { this.email = e;     return this; }
            Builder phone(String p)    { this.phone = p;     return this; }

            Person build() {
                // Validate here
                if (name == null || name.isEmpty()) throw new IllegalStateException("Name required");
                return new Person(this);
            }
        }
    }

    // Usage: Person p = new Person.Builder("Alice").age(30).email("a@b.com").build();

    // ── Factory Method ────────────────────────────────────────────────────────

    interface Notification {
        void send(String message);
    }

    static class EmailNotification implements Notification {
        public void send(String message) { System.out.println("Email: " + message); }
    }

    static class SMSNotification implements Notification {
        public void send(String message) { System.out.println("SMS: " + message); }
    }

    static class NotificationFactory {
        public static Notification create(String type) {
            return switch (type.toLowerCase()) {
                case "email" -> new EmailNotification();
                case "sms"   -> new SMSNotification();
                default      -> throw new IllegalArgumentException("Unknown type: " + type);
            };
        }
    }

    // ── Strategy Pattern ──────────────────────────────────────────────────────

    interface SortStrategy {
        void sort(int[] arr);
    }

    static class Sorter {
        private SortStrategy strategy;

        Sorter(SortStrategy strategy) { this.strategy = strategy; }

        void setStrategy(SortStrategy strategy) { this.strategy = strategy; }

        void sort(int[] arr) { strategy.sort(arr); }
    }

    // Usage with lambda (functional strategy):
    // Sorter s = new Sorter(arr -> Arrays.sort(arr));
    // s.setStrategy(arr -> { /* bubble sort */ });

    // ── Observer Pattern ──────────────────────────────────────────────────────

    interface EventListener<T> {
        void onEvent(T event);
    }

    static class EventBus<T> {
        private final List<EventListener<T>> listeners = new CopyOnWriteArrayList<>();

        public void subscribe(EventListener<T> listener)   { listeners.add(listener); }
        public void unsubscribe(EventListener<T> listener) { listeners.remove(listener); }

        public void publish(T event) {
            listeners.forEach(l -> l.onEvent(event));
        }
    }

    // ── Decorator Pattern ─────────────────────────────────────────────────────

    interface TextProcessor {
        String process(String text);
    }

    static class BaseProcessor implements TextProcessor {
        public String process(String text) { return text; }
    }

    static class TrimDecorator implements TextProcessor {
        private final TextProcessor wrapped;
        TrimDecorator(TextProcessor w) { this.wrapped = w; }
        public String process(String text) { return wrapped.process(text).trim(); }
    }

    static class UpperCaseDecorator implements TextProcessor {
        private final TextProcessor wrapped;
        UpperCaseDecorator(TextProcessor w) { this.wrapped = w; }
        public String process(String text) { return wrapped.process(text).toUpperCase(); }
    }

    // Usage: new UpperCaseDecorator(new TrimDecorator(new BaseProcessor())).process("  hello  ")

    // ── Proxy Pattern ─────────────────────────────────────────────────────────

    interface DataService {
        String getData(String key);
    }

    static class RealDataService implements DataService {
        public String getData(String key) { return "data_" + key; }
    }

    static class CachingProxy implements DataService {
        private final DataService real = new RealDataService();
        private final Map<String, String> cache = new HashMap<>();

        public String getData(String key) {
            return cache.computeIfAbsent(key, real::getData);
        }
    }
}


// ============================================================================
// SECTION 10: JAVA I/O AND NIO
// ============================================================================

class IOAndNIO {

    // ── Traditional I/O (Blocking) ────────────────────────────────────────────

    public void traditionalIO(String inputPath, String outputPath) throws IOException {
        // Buffered streams for efficiency (reads/writes in chunks)
        try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {

            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line.toUpperCase());
                writer.newLine();
            }
        }
    }

    // ── Java NIO (java.nio) ───────────────────────────────────────────────────

    public void nioFileOps() throws IOException {
        Path path = Path.of("data.txt");

        // Reading
        String content = Files.readString(path);            // Java 11+ (small files)
        List<String> lines = Files.readAllLines(path);      // All lines into list
        byte[] bytes = Files.readAllBytes(path);

        // Writing
        Files.writeString(path, "content", StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        Files.write(path, List.of("line1", "line2"));

        // Streaming large files
        try (Stream<String> lineStream = Files.lines(path)) {
            lineStream.filter(l -> !l.isEmpty()).forEach(System.out::println);
        }

        // Path operations
        Path parent = path.getParent();
        Path absolute = path.toAbsolutePath();
        boolean exists = Files.exists(path);
        Files.createDirectories(Path.of("a/b/c"));
        Files.copy(path, Path.of("backup.txt"), StandardCopyOption.REPLACE_EXISTING);
        Files.move(path, Path.of("moved.txt"));
        Files.delete(path);

        // Walk directory tree
        try (Stream<Path> walk = Files.walk(Path.of("."))) {
            walk.filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .forEach(System.out::println);
        }
    }

    // ── NIO Channels & Buffers ────────────────────────────────────────────────

    public void channelBufferDemo() throws IOException {
        // Buffer: fixed-size container with position, limit, capacity
        ByteBuffer buffer = ByteBuffer.allocate(1024);
        buffer.put("hello".getBytes());   // Write mode: position advances
        buffer.flip();                    // Switch to read mode: limit=position, position=0
        byte[] data = new byte[buffer.remaining()];
        buffer.get(data);                 // Read from buffer
        buffer.clear();                   // Reset for reuse

        // Channel I/O (non-blocking capable)
        try (FileChannel channel = FileChannel.open(Path.of("data.txt"),
                StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            ByteBuffer buf = ByteBuffer.allocate(1024);
            int bytesRead = channel.read(buf);
            buf.flip();
            channel.write(buf);

            // Memory-mapped file (very fast for large files)
            MappedByteBuffer mapped = channel.map(
                FileChannel.MapMode.READ_WRITE, 0, channel.size());
        }
    }

    // ── Serialization ─────────────────────────────────────────────────────────

    // serialVersionUID prevents InvalidClassException during deserialization
    static class SerializableDTO implements Serializable {
        private static final long serialVersionUID = 1L;

        private String name;
        private transient String password;  // transient: not serialized
        private int age;

        // readResolve: control what object is returned during deserialization
        // readObject: custom deserialization logic
        // writeObject: custom serialization logic

        private void writeObject(ObjectOutputStream out) throws IOException {
            out.defaultWriteObject();
            out.writeUTF("extra encrypted data");
        }

        private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
            in.defaultReadObject();
            String extra = in.readUTF();
        }
    }
}


// ============================================================================
// SECTION 11: JAVA RECORDS, SEALED CLASSES, PATTERN MATCHING (MODERN JAVA)
// ============================================================================

class ModernJava {

    // ── Records (Java 16+) ───────────────────────────────────────────────────

    // Compact, immutable data carriers. Auto-generates: constructor, getters, equals, hashCode, toString
    record Point(int x, int y) {
        // Compact canonical constructor (validation)
        Point {
            if (x < 0 || y < 0) throw new IllegalArgumentException("Coordinates must be positive");
        }

        // Custom methods allowed
        double distanceTo(Point other) {
            return Math.sqrt(Math.pow(x - other.x, 2) + Math.pow(y - other.y, 2));
        }
    }

    // ── Sealed Classes (Java 17+) ────────────────────────────────────────────

    // Restricts which classes can extend/implement (closed hierarchy)
    sealed interface Shape permits Circle, Rectangle3, Triangle {}

    record Circle(double radius) implements Shape {}
    record Rectangle3(double w, double h) implements Shape {}
    record Triangle(double base, double height) implements Shape {}

    // Pattern matching with sealed classes (exhaustive switch — no default needed!)
    double area(Shape shape) {
        return switch (shape) {
            case Circle c    -> Math.PI * c.radius() * c.radius();
            case Rectangle3 r -> r.w() * r.h();
            case Triangle t  -> 0.5 * t.base() * t.height();
            // No default needed — compiler verifies exhaustiveness!
        };
    }

    // ── Pattern Matching: instanceof (Java 16+) ───────────────────────────────

    public void patternMatchingDemo(Object obj) {
        // Old way
        if (obj instanceof String) {
            String s = (String) obj;
            System.out.println(s.length());
        }

        // New way — pattern variable
        if (obj instanceof String s && s.length() > 5) {
            System.out.println(s.toUpperCase()); // s is scoped here
        }
    }

    // ── Switch Expressions (Java 14+) ────────────────────────────────────────

    public String dayType(DayOfWeek day) {
        return switch (day) {
            case MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY -> "Weekday";
            case SATURDAY, SUNDAY -> "Weekend";
        };
    }

    // ── Text Blocks (Java 15+) ───────────────────────────────────────────────

    String json = """
            {
                "name": "Alice",
                "age": 30
            }
            """;

    // ── var (Java 10+) ───────────────────────────────────────────────────────

    public void varDemo() {
        var list = new ArrayList<String>();  // Compiler infers ArrayList<String>
        var map = new HashMap<String, List<Integer>>();
        // var can only be used for local variables (not fields, parameters, return types)
        for (var entry : map.entrySet()) {  // Works in for-each too
            System.out.println(entry.getKey());
        }
    }
}


// ============================================================================
// SECTION 12: JAVA REFLECTION & ANNOTATIONS
// ============================================================================

class ReflectionAndAnnotations {

    // ── Custom Annotations ────────────────────────────────────────────────────

    @interface Retry {
        int times() default 3;
        long delayMs() default 1000;
    }

    @interface Validated {}

    // ── Reflection API ────────────────────────────────────────────────────────

    @Retry(times = 5)
    public void reflectionDemo() throws Exception {
        Class<?> clazz = String.class;  // or Class.forName("java.lang.String")

        // Inspect fields
        Field[] fields = clazz.getDeclaredFields();
        for (Field f : fields) {
            f.setAccessible(true);  // Break private access
        }

        // Inspect methods
        Method[] methods = clazz.getDeclaredMethods();
        Method method = clazz.getDeclaredMethod("charAt", int.class);

        // Invoke method dynamically
        String instance = "hello";
        char ch = (char) method.invoke(instance, 0);  // 'h'

        // Create instance dynamically
        Constructor<?> ctor = String.class.getDeclaredConstructor(String.class);
        String newStr = (String) ctor.newInstance("world");

        // Read annotations
        Method myMethod = ReflectionAndAnnotations.class.getDeclaredMethod("reflectionDemo");
        if (myMethod.isAnnotationPresent(Retry.class)) {
            Retry retry = myMethod.getAnnotation(Retry.class);
            System.out.println("Retry times: " + retry.times());
        }

        // Generics via reflection (workaround for type erasure)
        Field listField = ArrayList.class.getDeclaredField("elementData");
        Type genericType = listField.getGenericType();
    }
}


// ============================================================================
// SECTION 13: COMMON ALGORITHMS & DATA STRUCTURES IN JAVA
// ============================================================================

class AlgorithmsAndDSA {

    // ── LRU Cache ─────────────────────────────────────────────────────────────

    static class LRUCache {
        private final int capacity;
        // LinkedHashMap with accessOrder=true evicts least-recently-accessed
        private final LinkedHashMap<Integer, Integer> cache;

        LRUCache(int capacity) {
            this.capacity = capacity;
            this.cache = new LinkedHashMap<>(capacity, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
                    return size() > capacity;
                }
            };
        }

        public int get(int key) {
            return cache.getOrDefault(key, -1);
        }

        public void put(int key, int value) {
            cache.put(key, value);
        }
    }

    // ── Trie ─────────────────────────────────────────────────────────────────

    static class Trie {
        private final TrieNode root = new TrieNode();

        static class TrieNode {
            TrieNode[] children = new TrieNode[26];
            boolean isEnd;
        }

        public void insert(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) node.children[idx] = new TrieNode();
                node = node.children[idx];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) return false;
                node = node.children[idx];
            }
            return node.isEnd;
        }

        public boolean startsWith(String prefix) {
            TrieNode node = root;
            for (char c : prefix.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) return false;
                node = node.children[idx];
            }
            return true;
        }
    }

    // ── Graph BFS & DFS ───────────────────────────────────────────────────────

    static class Graph {
        private final Map<Integer, List<Integer>> adjacency = new HashMap<>();

        public void addEdge(int u, int v) {
            adjacency.computeIfAbsent(u, k -> new ArrayList<>()).add(v);
            adjacency.computeIfAbsent(v, k -> new ArrayList<>()).add(u);
        }

        public List<Integer> bfs(int start) {
            List<Integer> result = new ArrayList<>();
            Set<Integer> visited = new HashSet<>();
            Queue<Integer> queue = new ArrayDeque<>();

            visited.add(start);
            queue.offer(start);

            while (!queue.isEmpty()) {
                int node = queue.poll();
                result.add(node);
                for (int neighbor : adjacency.getOrDefault(node, List.of())) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue.offer(neighbor);
                    }
                }
            }
            return result;
        }

        public List<Integer> dfs(int start) {
            List<Integer> result = new ArrayList<>();
            dfsHelper(start, new HashSet<>(), result);
            return result;
        }

        private void dfsHelper(int node, Set<Integer> visited, List<Integer> result) {
            visited.add(node);
            result.add(node);
            for (int neighbor : adjacency.getOrDefault(node, List.of())) {
                if (!visited.contains(neighbor)) {
                    dfsHelper(neighbor, visited, result);
                }
            }
        }
    }

    // ── Merge Sort ────────────────────────────────────────────────────────────

    public int[] mergeSort(int[] arr) {
        if (arr.length <= 1) return arr;
        int mid = arr.length / 2;
        int[] left  = mergeSort(Arrays.copyOfRange(arr, 0, mid));
        int[] right = mergeSort(Arrays.copyOfRange(arr, mid, arr.length));
        return merge(left, right);
    }

    private int[] merge(int[] left, int[] right) {
        int[] result = new int[left.length + right.length];
        int i = 0, j = 0, k = 0;
        while (i < left.length && j < right.length) {
            result[k++] = left[i] <= right[j] ? left[i++] : right[j++];
        }
        while (i < left.length)  result[k++] = left[i++];
        while (j < right.length) result[k++] = right[j++];
        return result;
    }

    // ── Two Pointers / Sliding Window ─────────────────────────────────────────

    // Max sum subarray of size k
    public int maxSubarraySum(int[] nums, int k) {
        int windowSum = 0;
        for (int i = 0; i < k; i++) windowSum += nums[i];
        int max = windowSum;
        for (int i = k; i < nums.length; i++) {
            windowSum += nums[i] - nums[i - k];
            max = Math.max(max, windowSum);
        }
        return max;
    }

    // Longest substring with at most k distinct characters
    public int longestSubstringKDistinct(String s, int k) {
        Map<Character, Integer> freq = new HashMap<>();
        int left = 0, max = 0;
        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);
            freq.merge(c, 1, Integer::sum);
            while (freq.size() > k) {
                char lc = s.charAt(left++);
                freq.merge(lc, -1, Integer::sum);
                if (freq.get(lc) == 0) freq.remove(lc);
            }
            max = Math.max(max, right - left + 1);
        }
        return max;
    }

    // ── Dynamic Programming ───────────────────────────────────────────────────

    // Coin Change (min coins to make amount)
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    // Longest Common Subsequence
    public int lcs(String a, String b) {
        int m = a.length(), n = b.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = a.charAt(i-1) == b.charAt(j-1)
                    ? dp[i-1][j-1] + 1
                    : Math.max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[m][n];
    }
}


// ============================================================================
// SECTION 14: INTERVIEW Q&A QUICK REFERENCE
// ============================================================================

/*
 * ============================================================================
 * TOP JAVA INTERVIEW QUESTIONS FOR 8+ YEARS EXPERIENCE
 * ============================================================================
 *
 * CORE JAVA
 * ─────────
 * Q: What is the difference between == and .equals()?
 *    == compares references (memory address) for objects; .equals() compares logical content.
 *    Always use .equals() for object comparison, and == only for primitives.
 *
 * Q: Why is String immutable in Java?
 *    Security (class loading, network connections), thread-safety, caching hashCode,
 *    and String Pool optimization (interning).
 *
 * Q: What is the contract between equals() and hashCode()?
 *    If a.equals(b) is true, then a.hashCode() == b.hashCode() must be true.
 *    Violation breaks HashMap, HashSet correctness.
 *
 * Q: What is the difference between abstract class and interface?
 *    Abstract class: single inheritance, can have state, constructors, access modifiers.
 *    Interface: multiple implementation, Java 8+ allows default/static methods.
 *
 * Q: What is Java's memory model?
 *    Heap (objects), Stack (frames), Metaspace (class metadata), PC register, Native stack.
 *    Young Gen (Eden + Survivors) + Old Gen (Tenured) + Metaspace.
 *
 * Q: What is the difference between checked and unchecked exceptions?
 *    Checked: must handle or declare (IOException). Unchecked: RuntimeException — optional handling.
 *
 * Q: How does HashMap work internally?
 *    Array of buckets. Key.hashCode() → bucket index. Collision: LinkedList (Java 7),
 *    LinkedList → TreeNode when bucket size >= 8 (Java 8). Resizes at load factor 0.75.
 *
 * Q: What is ConcurrentModificationException?
 *    Thrown when collection is modified while iterating via Iterator (non-concurrent collection).
 *    Fix: use Iterator.remove(), CopyOnWriteArrayList, or collect changes then apply.
 *
 * GENERICS & COLLECTIONS
 * ─────────────────────
 * Q: What is type erasure?
 *    Generics are compile-time only. At runtime, List<String> and List<Integer> are both List.
 *    JVM has no generic type info. This is why you can't do instanceof List<String>.
 *
 * Q: Explain PECS (Producer Extends, Consumer Super).
 *    Use ? extends T when you read (produce) from collection.
 *    Use ? super T when you write (consume) into collection.
 *
 * CONCURRENCY
 * ──────────
 * Q: What is the Java Memory Model (JMM)?
 *    Defines happens-before relationships. A volatile write happens-before the subsequent read.
 *    A monitor unlock happens-before subsequent lock on same monitor.
 *
 * Q: volatile vs synchronized?
 *    volatile: visibility guarantee only, no atomicity for compound operations.
 *    synchronized: both visibility AND atomicity, heavier weight.
 *
 * Q: What is a race condition?
 *    When multiple threads access shared state concurrently and result depends on scheduling.
 *    Fix: synchronization, atomic operations, or thread-safe data structures.
 *
 * Q: What is deadlock? How to prevent?
 *    Two threads each holding a lock the other needs. Prevention: consistent lock ordering,
 *    tryLock with timeout, lock fewer resources, use higher-level constructs.
 *
 * Q: Thread pool sizing?
 *    CPU-bound: N+1 threads (N = CPU cores).
 *    IO-bound: significantly more threads (depends on IO wait ratio).
 *    Use Little's Law: threads = throughput × latency.
 *
 * Q: CompletableFuture vs Future?
 *    Future: blocking .get(), no chaining.
 *    CompletableFuture: non-blocking, composable pipelines, exception handling.
 *
 * STREAMS & FUNCTIONAL
 * ────────────────────
 * Q: What is lazy evaluation in streams?
 *    Intermediate operations (filter, map) don't execute until terminal operation called.
 *    Short-circuit operations (findFirst, anyMatch) stop as soon as result determined.
 *
 * Q: parallel stream caveats?
 *    Uses ForkJoinPool.commonPool(). Overhead for small sets. Not always faster.
 *    Avoid shared mutable state, stateful operations, or IO in parallel streams.
 *
 * MODERN JAVA
 * ───────────
 * Q: When to use records?
 *    For immutable data transfer objects, value objects, tuples. Not for entities that change.
 *
 * Q: What are sealed classes for?
 *    To express closed algebraic data types. Enables exhaustive pattern matching in switch.
 *
 * GC & PERFORMANCE
 * ────────────────
 * Q: What triggers a Full GC?
 *    Old Gen full, promotion failure, System.gc() call, Metaspace full.
 *    Full GC is stop-the-world — avoid by tuning heap ratios and GC algorithm.
 *
 * Q: How to diagnose memory leaks in Java?
 *    Heap dumps (jmap -dump), profilers (VisualVM, JProfiler, async-profiler),
 *    GC logs (-Xlog:gc*), look for growing heap that doesn't recover after GC.
 *
 * Q: String.intern() — when to use?
 *    When creating many equal strings (e.g., from parsing), intern to save memory.
 *    Tradeoff: interned strings live in string pool (Metaspace) for lifetime of JVM.
 *
 * ============================================================================
 * KEY PERFORMANCE TIPS
 * ============================================================================
 *
 * 1.  Use StringBuilder in loops, not String concatenation.
 * 2.  Prefer ArrayList over LinkedList (better cache locality, less GC).
 * 3.  Initialize collections with expected capacity: new ArrayList<>(1000).
 * 4.  Use EnumSet/EnumMap for enum keys — fastest possible.
 * 5.  Avoid excessive autoboxing in hot paths; use primitive streams (IntStream).
 * 6.  Prefer ConcurrentHashMap over synchronizedMap — much better concurrency.
 * 7.  Use connection pooling (HikariCP) for DB; don't create connections per request.
 * 8.  Use lazy initialization for expensive singletons.
 * 9.  Prefer streams with early termination (findFirst, anyMatch) over full traversal.
 * 10. Use try-with-resources; always close I/O streams.
 * 11. Avoid finalizers — use Cleaner (Java 9+) or close() instead.
 * 12. Profile before optimizing — measure with JMH.
 *
 * ============================================================================
 * BEST PRACTICES CHECKLIST
 * ============================================================================
 *
 * [ ] Override both equals() AND hashCode() together.
 * [ ] Always make defensive copies for mutable fields in immutable classes.
 * [ ] Use Optional as return type, not as field/parameter.
 * [ ] Restore interrupted status: Thread.currentThread().interrupt() in catch.
 * [ ] Always release locks in finally blocks.
 * [ ] Never call wait() without while() loop (spurious wakeup protection).
 * [ ] Use specific exception types over generic Exception.
 * [ ] Chain exceptions to preserve original cause.
 * [ ] Prefer composition over inheritance (Effective Java).
 * [ ] Design for thread safety from the start — retroactive thread-safety is hard.
 * [ ] Use dependency injection over static factory or new.
 * [ ] Write to interfaces, not implementations (List, not ArrayList).
 * [ ] Close resources with try-with-resources (AutoCloseable).
 * [ ] Don't use raw types — always parameterize generics.
 * [ ] Use @Override annotation — catches subtle bugs at compile time.
 * [ ] Shutdown ExecutorService cleanly (shutdown + awaitTermination + shutdownNow).
 *
 * Good luck with your Java interview!
 */
