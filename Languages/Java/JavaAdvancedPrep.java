// ============================================================================
// JAVA INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
// For 8+ Years SDE Experience
// ============================================================================
// TOPICS COVERED:
//  1. JVM Internals & Memory Model
//  2. Core OOP (with all edge cases)
//  3. Generics (wildcards, bounds, type erasure)
//  4. Collections Framework (internals + pitfalls)
//  5. Concurrency & Multithreading (deep dive)
//  6. Java Memory Model & volatile/happens-before
//  7. Functional Programming (Streams, Lambdas, Optional)
//  8. Exception Handling (edge cases)
//  9. Design Patterns (with Java idioms)
// 10. Java 8-21 Modern Features
// 11. Garbage Collection (G1, ZGC, tuning)
// 12. Reflection & Annotations
// 13. I/O & NIO
// 14. Spring Framework Internals
// 15. Performance & Profiling
// ============================================================================

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.util.function.*;
import java.util.stream.*;
import java.lang.ref.*;
import java.lang.reflect.*;
import java.lang.annotation.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.nio.channels.*;
import java.util.Optional;
import java.time.*;
import java.util.WeakHashMap;

// ============================================================================
// SECTION 1: JVM INTERNALS & MEMORY MODEL
// ============================================================================

/**
 * JVM Memory Areas:
 * - Heap: Objects and arrays. Divided into Young Gen (Eden + S0 + S1) and Old Gen.
 * - Stack: Per-thread. Holds stack frames (local vars, operand stack, frame data).
 * - Method Area (Metaspace in Java 8+): Class metadata, static fields, constants.
 * - PC Register: Per-thread. Points to current bytecode instruction.
 * - Native Method Stack: Per-thread. Used for native (JNI) methods.
 *
 * INTERVIEWER TRAPS:
 * Q: Where are static variables stored?
 * A: In Metaspace (Method Area), NOT on the heap. But the OBJECTS they reference ARE on heap.
 *
 * Q: Where are String literals stored?
 * A: In the String Pool, which is on the heap since Java 7 (was in PermGen/Method Area before).
 *
 * Q: What happens when Metaspace fills up?
 * A: OutOfMemoryError: Metaspace (not PermGen since Java 8).
 */
class JVMInternals {

    // Stack vs Heap demonstration
    static int counter = 0; // Stored in Metaspace (static field reference in Metaspace, value on heap if it's an Object)

    public void memoryDemo() {
        int localVar = 42;          // Stack (primitive)
        String localStr = "hello";  // Stack reference -> String Pool (heap)
        Object obj = new Object();  // Stack reference -> Object on Heap

        // localVar lives and dies with this stack frame
        // obj on heap lives until GC collects (no more references)
    }

    // Class Loading: Bootstrap -> Extension/Platform -> Application ClassLoader
    // INTERVIEWER TRAP: Can you load same class twice?
    // YES, with different ClassLoaders -> they are considered DIFFERENT classes
    public void classLoaderDemo() throws Exception {
        ClassLoader cl1 = new URLClassLoader(new java.net.URL[]{});
        ClassLoader cl2 = new URLClassLoader(new java.net.URL[]{});
        // Classes loaded by cl1 != classes loaded by cl2 even if same bytecode
        // This is how OSGi, hot-reloading, and app servers work
    }

    // JIT Compilation levels:
    // Level 0: Interpreted
    // Level 1-3: C1 (client) compiler - fast compilation, less optimization
    // Level 4: C2 (server) compiler - slow compilation, heavy optimization
    // Tiered compilation (default since Java 8): starts C1, promotes hot methods to C2

    // String Pool internals
    public void stringPoolDemo() {
        String s1 = "hello";          // String Pool
        String s2 = "hello";          // Same String Pool reference
        String s3 = new String("hello"); // NEW heap object, NOT pool
        String s4 = s3.intern();      // Back to String Pool

        System.out.println(s1 == s2);  // true  (same pool reference)
        System.out.println(s1 == s3);  // false (s3 is heap object)
        System.out.println(s1 == s4);  // true  (intern returns pool reference)
        System.out.println(s1.equals(s3)); // true (content equal)

        // TRAP: String concatenation with + creates new objects (unless compile-time constants)
        String a = "hel";
        String b = "lo";
        String c = a + b;        // NOT in pool (runtime concat -> StringBuilder internally)
        System.out.println(s1 == c); // false
        System.out.println(s1 == "hel" + "lo"); // true (compile-time constant folding)
    }
}

// ============================================================================
// SECTION 2: CORE OOP - EDGE CASES INTERVIEWERS LOVE
// ============================================================================

/**
 * INHERITANCE, POLYMORPHISM, HIDING vs OVERRIDING
 */
class Animal {
    String type = "Animal";        // Field - can be HIDDEN, NOT overridden

    public String getType() {      // Instance method - can be OVERRIDDEN
        return "Animal";
    }

    public static String staticMethod() { // Static method - can be HIDDEN, NOT overridden
        return "Animal.staticMethod";
    }

    // Final method - CANNOT be overridden
    public final void finalMethod() {
        System.out.println("Animal.finalMethod");
    }
}

class Dog extends Animal {
    String type = "Dog";           // HIDES Animal.type (not override)

    @Override
    public String getType() {      // OVERRIDES Animal.getType (polymorphic)
        return "Dog";
    }

    public static String staticMethod() { // HIDES Animal.staticMethod (NOT polymorphic)
        return "Dog.staticMethod";
    }
}

class PolymorphismEdgeCases {
    public void demonstrate() {
        Animal a = new Dog();

        // INSTANCE METHOD: Polymorphism applies -> calls Dog.getType()
        System.out.println(a.getType());          // "Dog"

        // FIELD: NO polymorphism -> Animal.type (reference type determines field)
        System.out.println(a.type);               // "Animal"  ← INTERVIEWER TRAP

        // STATIC METHOD: NO polymorphism -> Animal.staticMethod (reference type determines)
        System.out.println(Animal.staticMethod()); // "Animal.staticMethod"
        System.out.println(Dog.staticMethod());    // "Dog.staticMethod"
        System.out.println(a.staticMethod());      // "Animal.staticMethod" ← TRAP (avoid calling statics on instances)

        Dog d = (Dog) a;
        System.out.println(d.type);               // "Dog" (reference is Dog)
    }
}

/**
 * CONSTRUCTOR CHAINING & INITIALIZATION ORDER
 * INTERVIEWER LOVES THIS: What is the order of initialization?
 * Order: static fields/blocks (parent first) -> instance fields/blocks -> constructor
 */
class Parent {
    static int staticField = initStatic("Parent.staticField"); // 1st
    int instanceField = initInstance("Parent.instanceField");  // 3rd

    static {
        System.out.println("Parent static block");             // 2nd
    }

    {
        System.out.println("Parent instance block");           // 4th
    }

    Parent() {
        System.out.println("Parent constructor");              // 5th
    }

    static int initStatic(String name) {
        System.out.println("Init: " + name);
        return 0;
    }

    int initInstance(String name) {
        System.out.println("Init: " + name);
        return 0;
    }
}

class Child extends Parent {
    static int staticField = initStatic("Child.staticField");  // 6th (child static)
    int instanceField = initInstance("Child.instanceField");   // 8th

    static {
        System.out.println("Child static block");              // 7th
    }

    {
        System.out.println("Child instance block");            // 9th
    }

    Child() {
        super(); // Implicitly called first! 
        // super() is ALWAYS first in constructor. Can't access 'this' before super()
        System.out.println("Child constructor");               // 10th
    }
}

/**
 * INTERFACE vs ABSTRACT CLASS - Deep Dive
 * TRAP: Can interface have state? Yes, since Java 8 (default/static methods, private methods Java 9+)
 */
interface Flyable {
    int MAX_ALTITUDE = 10000; // implicitly public static final

    void fly();              // implicitly public abstract

    default void land() {   // Java 8+: default method (has body)
        System.out.println("Landing...");
    }

    static void info() {    // Java 8+: static method in interface
        System.out.println("Flyable interface");
    }

    private void helper() { // Java 9+: private method (used by default methods)
        System.out.println("Helper");
    }
}

interface Swimmable {
    default void land() {   // CONFLICT with Flyable.land()
        System.out.println("Amphibious landing...");
    }
}

// MUST override land() to resolve diamond problem with default methods
class Duck implements Flyable, Swimmable {
    @Override
    public void fly() { System.out.println("Duck flying"); }

    @Override
    public void land() {                     // MANDATORY override to resolve conflict
        Flyable.super.land();                // Can choose which default to call
        // Swimmable.super.land();           // Or this one
    }
}

/**
 * COVARIANT RETURN TYPE & GENERICS INTERACTION
 */
class BaseFactory {
    public Animal create() { return new Animal(); }
}

class DogFactory extends BaseFactory {
    @Override
    public Dog create() { return new Dog(); } // Covariant return type (Java 5+)
}

/**
 * equals() and hashCode() CONTRACT
 * CRITICAL: If equals() is overridden, hashCode() MUST be overridden
 * If a.equals(b) -> a.hashCode() == b.hashCode() (mandatory)
 * If a.hashCode() == b.hashCode() -> a.equals(b) may be false (allowed, but bad for performance)
 */
class Employee {
    private final String id;
    private String name;

    Employee(String id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;               // Reflexive
        if (!(o instanceof Employee)) return false; // null-safe, handles subclasses
        Employee e = (Employee) o;
        return Objects.equals(id, e.id);          // Only compare id (identity)
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);                  // MUST use same field(s) as equals
    }

    // TRAP: What if you put Employee in HashSet, then change the name field?
    // If name was in equals/hashCode, the object would be "lost" in the set!
    // LESSON: Use IMMUTABLE fields in equals/hashCode
}

/**
 * IMMUTABILITY - How to make a truly immutable class
 * TRAP: Shallow vs Deep immutability
 */
final class ImmutablePerson {          // final: cannot be subclassed
    private final String name;          // final: reference cannot change
    private final List<String> hobbies; // TRAP: final reference, but List is mutable!

    public ImmutablePerson(String name, List<String> hobbies) {
        this.name = name;
        // Defensive copy on construction
        this.hobbies = Collections.unmodifiableList(new ArrayList<>(hobbies));
    }

    public String getName() { return name; } // String is immutable, safe to return

    public List<String> getHobbies() {
        return hobbies; // unmodifiableList wrapper prevents modification
        // Alternative: return Collections.unmodifiableList(hobbies);
        // Alternative: return new ArrayList<>(hobbies); // defensive copy
    }

    // NO setters
    // NO public mutable fields
    // Class is final (or all constructors private + factory methods)
}

// ============================================================================
// SECTION 3: GENERICS - TYPE ERASURE, WILDCARDS, BOUNDS
// ============================================================================

/**
 * TYPE ERASURE: Generic type info is ERASED at compile time
 * At runtime, List<String> and List<Integer> are both just List
 * TRAP: You CANNOT do: new T(), T.class, instanceof T, new T[]
 */
class TypeErasureDemo<T> {

    // T is erased at runtime -> becomes Object (or bound type)
    private T value;

    // CANNOT do: return new T(); at runtime T is unknown
    // Workaround: pass Class<T> or use Supplier<T>
    public static <T> T createInstance(Class<T> clazz) throws Exception {
        return clazz.getDeclaredConstructor().newInstance(); // Reflection workaround
    }

    // TRAP: Overloading with generics doesn't work if erasure makes signatures identical
    // public void process(List<String> list) {} // Compile error:
    // public void process(List<Integer> list) {} // Both erase to process(List)

    // TRAP: Generic array creation
    // T[] arr = new T[10]; // COMPILE ERROR
    @SuppressWarnings("unchecked")
    T[] createArray(int size) {
        return (T[]) new Object[size]; // Only safe if array doesn't escape class
    }
}

/**
 * WILDCARDS: ? extends T (upper bound) vs ? super T (lower bound)
 * PECS: Producer Extends, Consumer Super
 *
 * ? extends T: You can READ T from it, CANNOT write (except null)
 * ? super T:   You can WRITE T to it, can only READ Object
 */
class WildcardsDemo {

    // PRODUCER: reads from source (upper bound)
    public static double sumList(List<? extends Number> list) {
        double sum = 0;
        for (Number n : list) { // Reading is safe - all are Number
            sum += n.doubleValue();
        }
        // list.add(1.0); // COMPILE ERROR - can't add (type safety)
        return sum;
    }

    // CONSUMER: writes to dest (lower bound)
    public static void addNumbers(List<? super Integer> list) {
        list.add(1);   // Safe: any list of Integer or supertype accepts Integer
        list.add(2);
        // Integer i = list.get(0); // COMPILE ERROR - might be List<Number> or List<Object>
        Object o = list.get(0);     // Only Object is safe to read
    }

    // PECS example: copy from src to dest
    public static <T> void copy(List<? extends T> src, List<? super T> dest) {
        for (T item : src) dest.add(item);
    }

    // Unbounded wildcard: when you only use Object methods
    public static void printList(List<?> list) {
        for (Object o : list) System.out.println(o);
        // list.add("x"); // COMPILE ERROR
    }

    // TRAP: List<Object> is NOT a supertype of List<String>
    // But List<?> IS a supertype of any List<X>
    public void trap() {
        List<String> strings = new ArrayList<>();
        // List<Object> objects = strings; // COMPILE ERROR - not covariant
        List<?> wildcard = strings;        // OK - wildcard accepts any
    }
}

/**
 * BOUNDED TYPE PARAMETERS
 */
class BoundsDemo {
    // Upper bound on type parameter
    public static <T extends Comparable<T>> T max(T a, T b) {
        return a.compareTo(b) >= 0 ? a : b;
    }

    // Multiple bounds
    public static <T extends Comparable<T> & Cloneable> T cloneableMax(T a, T b) {
        return a.compareTo(b) >= 0 ? a : b;
    }

    // TRAP: Type parameter vs wildcard
    // Use type parameter when you need to refer to the type elsewhere
    // Use wildcard when you don't care about the specific type
    public <T> void swap(List<T> list, int i, int j) {
        T temp = list.get(i);  // T lets you hold the element
        list.set(i, list.get(j));
        list.set(j, temp);
    }

    public void swapWildcard(List<?> list, int i, int j) {
        // helper needed because ? can't be stored and set back
        swapHelper(list, i, j);
    }
    private <T> void swapHelper(List<T> list, int i, int j) {
        T temp = list.get(i);
        list.set(i, list.get(j));
        list.set(j, temp);
    }
}

// ============================================================================
// SECTION 4: COLLECTIONS FRAMEWORK - INTERNALS & PITFALLS
// ============================================================================

/**
 * HASHMAP INTERNAL MECHANICS
 * - Array of buckets (Node[] table), default capacity 16, load factor 0.75
 * - index = hash(key) & (capacity - 1)  [bitwise AND instead of modulo - faster]
 * - Collision: LinkedList in bucket (Java 7-), TreeNode if bucket size >= 8 (Java 8+)
 * - TreeNode threshold: 8 to tree, 6 to unlink (TREEIFY_THRESHOLD / UNTREEIFY_THRESHOLD)
 * - Resize: when size > capacity * loadFactor, double capacity, rehash all
 */
class HashMapInternals {

    public void demonstratePitfalls() {

        // TRAP 1: Mutable key in HashMap
        List<String> mutableKey = new ArrayList<>(Arrays.asList("a", "b"));
        Map<List<String>, String> map = new HashMap<>();
        map.put(mutableKey, "value");
        mutableKey.add("c"); // Mutate the key!
        System.out.println(map.get(mutableKey)); // null! hashCode changed, can't find it

        // TRAP 2: HashMap is NOT thread-safe
        // Use ConcurrentHashMap or Collections.synchronizedMap()
        Map<String, Integer> unsafeMap = new HashMap<>();
        // Multiple threads modifying -> ConcurrentModificationException or data corruption

        // TRAP 3: Initial capacity tuning
        // If you know you'll have N entries: capacity = N / 0.75 + 1 (avoid resizing)
        Map<String, String> optimized = new HashMap<>(100); // No resize up to 75 entries

        // TRAP 4: null keys/values
        HashMap<String, String> hashMap = new HashMap<>();
        hashMap.put(null, "nullKey");    // HashMap allows ONE null key
        hashMap.put("key", null);        // HashMap allows null values
        // Hashtable does NOT allow null keys or values (legacy, use HashMap or ConcurrentHashMap)
    }

    // HashMap vs LinkedHashMap vs TreeMap
    public void mapVariants() {
        // HashMap: O(1) avg get/put, NO order guarantee
        Map<String, Integer> hashMap = new HashMap<>();

        // LinkedHashMap: O(1) avg, insertion-order or access-order
        Map<String, Integer> linkedMap = new LinkedHashMap<>(16, 0.75f, true); // access-order=true -> LRU cache!

        // TreeMap: O(log n) get/put, keys sorted (natural order or Comparator)
        Map<String, Integer> treeMap = new TreeMap<>();

        // LRU Cache using LinkedHashMap
        int capacity = 100;
        Map<String, Integer> lruCache = new LinkedHashMap<>(capacity, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, Integer> eldest) {
                return size() > capacity; // Auto-evict eldest when over capacity
            }
        };
    }
}

/**
 * ITERATOR & CONCURRENTMODIFICATIONEXCEPTION
 */
class IteratorPitfalls {

    public void failFastVsFailSafe() {
        List<String> list = new ArrayList<>(Arrays.asList("a", "b", "c"));

        // FAIL-FAST: ArrayList iterator checks modCount - throws ConcurrentModificationException
        try {
            for (String s : list) {
                if (s.equals("b")) list.remove(s); // THROWS ConcurrentModificationException
            }
        } catch (ConcurrentModificationException e) {
            System.out.println("Caught CME");
        }

        // CORRECT: Use iterator.remove()
        Iterator<String> it = list.iterator();
        while (it.hasNext()) {
            if (it.next().equals("b")) it.remove(); // Safe!
        }

        // CORRECT: Use removeIf() (Java 8+)
        list.removeIf(s -> s.equals("b"));

        // FAIL-SAFE: CopyOnWriteArrayList - iterates on snapshot
        List<String> cowList = new CopyOnWriteArrayList<>(Arrays.asList("a", "b", "c"));
        for (String s : cowList) {
            if (s.equals("b")) cowList.remove(s); // No CME, but iterates old snapshot
        }
        // Trade-off: CopyOnWriteArrayList copies entire array on every write -> expensive writes
    }

    // TRAP: for-each vs indexed loop removal
    public void indexedLoopTrap() {
        List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        // Removing while iterating forward - skips elements!
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) % 2 == 0) {
                list.remove(i); // After remove, next element shifts to current index
                // i++  makes us skip it!
            }
        }
        // Correct: iterate backwards
        for (int i = list.size() - 1; i >= 0; i--) {
            if (list.get(i) % 2 == 0) list.remove(i); // Safe
        }
    }
}

/**
 * COMPARABLE vs COMPARATOR
 */
class SortingDeepDive {

    // Comparable: natural ordering, implemented by the class itself
    static class Student implements Comparable<Student> {
        String name;
        int gpa;

        Student(String name, int gpa) { this.name = name; this.gpa = gpa; }

        @Override
        public int compareTo(Student other) {
            // TRAP: Don't use subtraction for int comparison (overflow risk with Integer.MIN_VALUE)
            // return this.gpa - other.gpa; // BUG if values near Integer.MIN/MAX_VALUE
            return Integer.compare(this.gpa, other.gpa); // CORRECT
        }
    }

    // Comparator: external ordering, flexible, chainable (Java 8+)
    public void comparatorDemo() {
        List<Student> students = new ArrayList<>();

        // Method reference + chained comparators
        students.sort(Comparator.comparingInt((Student s) -> s.gpa)
                                .reversed()
                                .thenComparing(s -> s.name));

        // Null handling
        Comparator<String> nullSafe = Comparator.nullsFirst(Comparator.naturalOrder());
        List<String> withNulls = Arrays.asList("b", null, "a");
        withNulls.sort(nullSafe);
    }

    // TRAP: TreeSet/TreeMap use compareTo for equality, NOT equals()!
    public void treeSetTrap() {
        TreeSet<Student> set = new TreeSet<>();
        Student s1 = new Student("Alice", 90);
        Student s2 = new Student("Bob", 90); // Same GPA
        set.add(s1);
        set.add(s2);
        System.out.println(set.size()); // 1! compareTo returns 0 -> considered EQUAL by TreeSet
        // Even though equals() might say they're different
        // FIX: Make compareTo consistent with equals(), or add secondary sort key
    }
}

// ============================================================================
// SECTION 5: CONCURRENCY & MULTITHREADING - DEEP DIVE
// ============================================================================

/**
 * THREAD LIFECYCLE:
 * NEW -> RUNNABLE -> RUNNING -> (BLOCKED/WAITING/TIMED_WAITING) -> TERMINATED
 *
 * BLOCKED: Waiting to acquire a monitor lock
 * WAITING: Indefinite wait (wait(), join(), LockSupport.park())
 * TIMED_WAITING: wait(timeout), sleep(), join(timeout)
 */
class ThreadDeepDive {

    // Thread creation options
    public void threadCreation() {
        // Option 1: Extend Thread (tight coupling, can't extend anything else)
        Thread t1 = new Thread() {
            @Override public void run() { System.out.println("Thread subclass"); }
        };

        // Option 2: Implement Runnable (preferred - separates task from execution)
        Thread t2 = new Thread(() -> System.out.println("Runnable lambda"));

        // Option 3: Callable + Future (returns result, can throw checked exceptions)
        ExecutorService executor = Executors.newFixedThreadPool(4);
        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(100);
            return 42;
        });

        // Option 4: CompletableFuture (non-blocking, composable)
        CompletableFuture<Integer> cf = CompletableFuture.supplyAsync(() -> 42);

        // TRAP: Thread.start() vs Thread.run()
        // start(): Creates new thread, calls run() asynchronously
        // run():   Executes run() in CURRENT thread (no new thread created!)
        t1.start(); // Correct
        // t1.run(); // Bug: runs synchronously in main thread

        // TRAP: Can you restart a thread?
        // NO! After thread terminates, calling start() again throws IllegalThreadStateException
    }

    // DAEMON vs USER threads
    public void daemonThreads() {
        Thread daemon = new Thread(() -> {
            while (true) {
                try { Thread.sleep(1000); } catch (InterruptedException e) { break; }
                System.out.println("Daemon running");
            }
        });
        daemon.setDaemon(true); // MUST set before start()
        daemon.start();
        // JVM exits when ALL USER threads finish, regardless of daemon threads running
        // Daemon threads don't prevent JVM shutdown
    }

    // SYNCHRONIZED - intrinsic lock (monitor)
    class SynchronizedDemo {
        private int count = 0;
        private final Object lock = new Object();

        // Synchronized instance method: locks on 'this'
        public synchronized void increment() { count++; }

        // Synchronized static method: locks on Class object
        public static synchronized void staticMethod() {}

        // Synchronized block: more granular, explicit lock object
        public void blockSync() {
            synchronized (lock) {
                count++; // Only this section is locked, not entire method
            }
        }

        // TRAP: Two different synchronized methods can run concurrently?
        // NO! Both lock on 'this'. Only ONE synchronized method per instance at a time.

        // TRAP: Synchronizing on 'this' vs private lock
        // External code can also sync on 'this', potentially causing deadlocks
        // Always prefer private final lock object
    }

    // DEADLOCK, LIVELOCK, STARVATION
    class DeadlockDemo {
        final Object lockA = new Object();
        final Object lockB = new Object();

        void thread1() {
            synchronized (lockA) {           // Acquires A
                try { Thread.sleep(100); } catch (InterruptedException e) {}
                synchronized (lockB) {       // Waits for B (held by thread2)
                    System.out.println("Thread1 done");
                }
            }
        }

        void thread2() {
            synchronized (lockB) {           // Acquires B
                try { Thread.sleep(100); } catch (InterruptedException e) {}
                synchronized (lockA) {       // Waits for A (held by thread1) -> DEADLOCK
                    System.out.println("Thread2 done");
                }
            }
        }

        // PREVENTION: Always acquire locks in same order
        void safeThread1() {
            synchronized (lockA) { synchronized (lockB) { /* work */ } }
        }
        void safeThread2() {
            synchronized (lockA) { synchronized (lockB) { /* work */ } } // Same order
        }

        // Or use tryLock() with timeout (ReentrantLock)
        ReentrantLock rl1 = new ReentrantLock();
        ReentrantLock rl2 = new ReentrantLock();

        void tryLockApproach() throws InterruptedException {
            while (true) {
                if (rl1.tryLock(100, TimeUnit.MILLISECONDS)) {
                    try {
                        if (rl2.tryLock(100, TimeUnit.MILLISECONDS)) {
                            try {
                                // Do work
                                return;
                            } finally { rl2.unlock(); }
                        }
                    } finally { rl1.unlock(); }
                }
                Thread.sleep(50); // Back-off before retry
            }
        }
    }
}

/**
 * JAVA MEMORY MODEL (JMM) - volatile, happens-before
 *
 * VISIBILITY PROBLEM: Without synchronization, changes by Thread A may NOT be
 * visible to Thread B (CPU caches, compiler reordering).
 *
 * HAPPENS-BEFORE relationships guarantee visibility and ordering:
 * 1. Program order (within single thread)
 * 2. Monitor unlock happens-before subsequent lock (on same monitor)
 * 3. volatile write happens-before subsequent volatile read (same variable)
 * 4. Thread.start() happens-before actions in the started thread
 * 5. All actions in thread happen-before Thread.join() returns
 * 6. Thread interruption happens-before interrupted thread detects it
 */
class JavaMemoryModelDemo {

    // VOLATILE: guarantees visibility, prevents instruction reordering around it
    // Does NOT make compound operations atomic (i++ is still NOT atomic with volatile)
    volatile boolean stopFlag = false;

    void worker() {
        while (!stopFlag) { // Reads latest value from memory, not CPU cache
            // do work
        }
    }

    void stop() {
        stopFlag = true; // Write is immediately visible to all threads
    }

    // Classic double-checked locking (broken without volatile pre-Java 5)
    private volatile Object instance; // volatile is ESSENTIAL here

    public Object getInstance() {
        if (instance == null) {                    // Check 1: avoid lock if initialized
            synchronized (this) {
                if (instance == null) {            // Check 2: avoid double init
                    instance = new Object();       // Without volatile: partially initialized
                    // object reference can be published before constructor completes!
                }
            }
        }
        return instance;
    }

    // TRAP: volatile vs synchronized
    // volatile: visibility only, no atomicity for compound ops
    // synchronized: visibility + atomicity + mutual exclusion

    // AtomicInteger for lock-free atomic operations (CAS - Compare And Swap)
    AtomicInteger atomicCount = new AtomicInteger(0);

    void atomicDemo() {
        atomicCount.incrementAndGet();                         // Atomic i++
        atomicCount.compareAndSet(5, 10);                     // CAS: if value==5, set to 10
        atomicCount.updateAndGet(v -> v * 2);                 // Atomic update function
        atomicCount.accumulateAndGet(5, Integer::sum);        // Atomic accumulate

        // LongAdder: better than AtomicLong for high-contention counters
        // Splits into multiple cells, reduces contention, merges on read
        LongAdder adder = new LongAdder();
        adder.increment();
        long total = adder.sum(); // Not instantaneously consistent under contention, but fast
    }
}

/**
 * EXECUTOR FRAMEWORK & THREAD POOLS
 */
class ExecutorFrameworkDeepDive {

    // ThreadPoolExecutor parameters (know all 7!)
    public void threadPoolParams() {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            4,                          // corePoolSize: threads kept alive even idle
            8,                          // maximumPoolSize: max threads ever
            60L, TimeUnit.SECONDS,      // keepAliveTime: idle non-core thread survives
            new ArrayBlockingQueue<>(100), // workQueue: holds tasks when all core threads busy
            Executors.defaultThreadFactory(), // threadFactory
            new ThreadPoolExecutor.CallerRunsPolicy() // rejectionPolicy: what to do when queue full
            // AbortPolicy: throws RejectedExecutionException (default)
            // DiscardPolicy: silently drops task
            // DiscardOldestPolicy: drops oldest queued task, retries
            // CallerRunsPolicy: runs task in calling thread (back-pressure)
        );

        // TRAP: Executors.newFixedThreadPool uses UNBOUNDED queue (LinkedBlockingQueue)
        // -> OOM if tasks submitted faster than consumed
        // Executors.newCachedThreadPool uses SynchronousQueue + unbounded threads
        // -> Thread explosion if tasks submitted faster than consumed

        // ALWAYS use explicit ThreadPoolExecutor with bounded queue in production!
    }

    // COMPLETABLEFUTURE - Java 8+ asynchronous composition
    public void completableFutureDemo() throws Exception {
        // Create
        CompletableFuture<String> cf1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> cf2 = CompletableFuture.supplyAsync(() -> "World");

        // Chain (thenApply: sync transform, thenApplyAsync: async transform)
        CompletableFuture<String> upper = cf1.thenApply(String::toUpperCase);

        // Combine two futures
        CompletableFuture<String> combined = cf1.thenCombine(cf2,
            (s1, s2) -> s1 + " " + s2);

        // Wait for all
        CompletableFuture<Void> allDone = CompletableFuture.allOf(cf1, cf2);

        // Wait for first
        CompletableFuture<Object> anyDone = CompletableFuture.anyOf(cf1, cf2);

        // Exception handling
        CompletableFuture<String> withFallback = cf1
            .thenApply(s -> { throw new RuntimeException("oops"); })
            .exceptionally(ex -> "fallback")
            .handle((result, ex) -> ex != null ? "error: " + ex.getMessage() : result);

        // TRAP: thenApply vs thenCompose (flatMap equivalent)
        CompletableFuture<CompletableFuture<String>> nested =
            cf1.thenApply(s -> CompletableFuture.supplyAsync(() -> s + "!")); // CF<CF<String>>

        CompletableFuture<String> flat =
            cf1.thenCompose(s -> CompletableFuture.supplyAsync(() -> s + "!")); // CF<String>

        // TRAP: Default executor is ForkJoinPool.commonPool()
        // For blocking ops, use a dedicated thread pool:
        ExecutorService blockingPool = Executors.newFixedThreadPool(10);
        CompletableFuture<String> onCustomPool =
            CompletableFuture.supplyAsync(() -> "result", blockingPool);
    }

    // LOCK vs SYNCHRONIZED
    public void lockDemo() throws InterruptedException {
        ReentrantLock lock = new ReentrantLock(true); // fair=true: threads served in order

        lock.lock();
        try {
            // critical section
        } finally {
            lock.unlock(); // MUST unlock in finally!
        }

        // tryLock with timeout (impossible with synchronized)
        if (lock.tryLock(100, TimeUnit.MILLISECONDS)) {
            try { /* work */ } finally { lock.unlock(); }
        }

        // Condition variables (like wait/notify but per-condition)
        Condition notEmpty = lock.newCondition();
        Condition notFull = lock.newCondition();
        // notEmpty.await() / notEmpty.signal() - more targeted than notifyAll()

        // ReadWriteLock: multiple readers OR one writer
        ReadWriteLock rwLock = new ReentrantReadWriteLock();
        rwLock.readLock().lock();   // Multiple threads can hold readLock simultaneously
        rwLock.readLock().unlock();
        rwLock.writeLock().lock();  // Exclusive: blocks all readers and writers
        rwLock.writeLock().unlock();

        // StampedLock (Java 8+): optimistic reads (even faster)
        StampedLock sl = new StampedLock();
        long stamp = sl.tryOptimisticRead(); // No lock, just a stamp
        // read data
        if (!sl.validate(stamp)) {          // Check if write happened since stamp
            stamp = sl.readLock();          // Fall back to pessimistic read
            try { /* re-read data */ } finally { sl.unlockRead(stamp); }
        }
    }
}

// ============================================================================
// SECTION 6: FUNCTIONAL PROGRAMMING - STREAMS, LAMBDAS, OPTIONAL
// ============================================================================

/**
 * STREAMS - Lazy evaluation, pipeline, terminal operations
 */
class StreamsDeepDive {

    public void streamPipelines() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        // LAZY: intermediate ops do nothing until terminal op is called
        Stream<Integer> lazyStream = numbers.stream()
            .filter(n -> { System.out.println("filter: " + n); return n > 5; })  // lazy
            .map(n -> { System.out.println("map: " + n); return n * 2; });        // lazy
        // Nothing printed yet!

        List<Integer> result = lazyStream.collect(Collectors.toList()); // NOW it executes

        // Short-circuit: stops early when possible
        Optional<Integer> first = numbers.stream()
            .filter(n -> n > 5)
            .findFirst(); // Only processes until first match, then stops

        // TRAP: Stream can only be consumed ONCE
        Stream<Integer> stream = numbers.stream();
        stream.forEach(System.out::println);
        // stream.forEach(System.out::println); // IllegalStateException: stream already operated upon
    }

    public void collectors() {
        List<String> words = Arrays.asList("apple", "banana", "cherry", "avocado");

        // Grouping
        Map<Character, List<String>> byFirstLetter =
            words.stream().collect(Collectors.groupingBy(w -> w.charAt(0)));

        // Counting per group
        Map<Character, Long> countByLetter =
            words.stream().collect(Collectors.groupingBy(w -> w.charAt(0), Collectors.counting()));

        // Joining
        String joined = words.stream().collect(Collectors.joining(", ", "[", "]"));

        // Partitioning (by predicate -> two groups: true/false)
        Map<Boolean, List<String>> longShort =
            words.stream().collect(Collectors.partitioningBy(w -> w.length() > 5));

        // toMap - TRAP: duplicate keys throw IllegalStateException!
        // Use merge function to handle duplicates
        Map<Integer, String> lengthToWord = words.stream()
            .collect(Collectors.toMap(
                String::length,
                w -> w,
                (existing, newVal) -> existing + "," + newVal // merge duplicates
            ));

        // flatMap: flatten nested collections
        List<List<Integer>> nested = Arrays.asList(
            Arrays.asList(1, 2), Arrays.asList(3, 4));
        List<Integer> flat = nested.stream()
            .flatMap(Collection::stream)
            .collect(Collectors.toList()); // [1, 2, 3, 4]

        // reduce
        int sum = numbers.stream().reduce(0, Integer::sum); // identity + accumulator
        Optional<Integer> product = numbers.stream().reduce((a, b) -> a * b); // no identity
    }

    // Parallel streams - TRAP: Not always faster!
    public void parallelStreams() {
        List<Integer> bigList = IntStream.range(0, 1_000_000).boxed().collect(Collectors.toList());

        long sum = bigList.parallelStream()
            .mapToLong(Integer::longValue)
            .sum(); // ForkJoinPool.commonPool() by default

        // TRAPS with parallel streams:
        // 1. Stateful operations (sorted, distinct, limit) reduce parallelism benefit
        // 2. Shared mutable state = data races
        // 3. Small datasets: overhead of parallelism > benefit
        // 4. Order-sensitive operations on unordered sources

        // BAD: shared mutable state in parallel stream
        List<Integer> bad = new ArrayList<>();
        bigList.parallelStream().filter(n -> n % 2 == 0)
               .forEach(bad::add); // Race condition! Elements may be missing or duplicated

        // GOOD: use collect
        List<Integer> good = bigList.parallelStream()
            .filter(n -> n % 2 == 0)
            .collect(Collectors.toList()); // Thread-safe

        // Custom ForkJoinPool for parallel stream
        ForkJoinPool customPool = new ForkJoinPool(4);
        try {
            long result2 = customPool.submit(
                () -> bigList.parallelStream().mapToLong(Integer::longValue).sum()
            ).get();
        } catch (Exception e) { e.printStackTrace(); }
    }

    // OPTIONAL - proper usage
    public void optionalUsage() {
        Optional<String> opt = Optional.of("hello");
        Optional<String> empty = Optional.empty();
        Optional<String> nullable = Optional.ofNullable(null); // Safe for null

        // TRAP: opt.get() on empty throws NoSuchElementException
        // TRAP: Optional.of(null) throws NullPointerException

        // Correct usage
        String value1 = opt.orElse("default");                    // value or default
        String value2 = opt.orElseGet(() -> computeDefault());    // lazy default
        String value3 = opt.orElseThrow(() -> new RuntimeException("missing")); // or throw

        // Chaining
        Optional<Integer> length = opt
            .filter(s -> s.length() > 3)
            .map(String::length);

        // flatMap when mapping returns Optional
        Optional<String> upper = opt.flatMap(s ->
            s.isEmpty() ? Optional.empty() : Optional.of(s.toUpperCase()));

        // ifPresent / ifPresentOrElse (Java 9+)
        opt.ifPresent(System.out::println);
        opt.ifPresentOrElse(System.out::println, () -> System.out.println("empty"));

        // TRAP: Do NOT use Optional as method parameter or field (use for return values only)
        // Reason: Optional has no serialization support, bad for beans/JPA

        // TRAP: Do NOT do opt.isPresent() + opt.get() - use orElse/map/ifPresent instead
    }

    private String computeDefault() { return "default"; }
    private List<Integer> numbers = Arrays.asList(1,2,3,4,5,6,7,8,9,10);
}

// ============================================================================
// SECTION 7: EXCEPTION HANDLING - EDGE CASES
// ============================================================================

/**
 * CHECKED vs UNCHECKED exceptions
 * Checked: extends Exception (not RuntimeException) - MUST handle or declare
 * Unchecked: extends RuntimeException or Error - NOT required to handle
 *
 * CONTROVERSIAL: Many argue checked exceptions are a design mistake
 * (Kotlin, C# dropped them; Spring wraps them as unchecked)
 */
class ExceptionEdgeCases {

    // TRAP: Exception in finally block MASKS exception from try block!
    public void finallyTrap() throws Exception {
        try {
            throw new RuntimeException("try exception");
        } finally {
            throw new RuntimeException("finally exception"); // MASKS try exception!
            // Original "try exception" is LOST
        }
    }

    // TRAP: return in finally overrides return in try
    public int returnInFinally() {
        try {
            return 1; // This return is IGNORED
        } finally {
            return 2; // This return wins
        }
    }

    // Try-with-resources: AutoCloseable, close() called in reverse order
    public void tryWithResources() {
        // Resources closed in REVERSE order: r2 closed first, then r1
        try (Resource r1 = new Resource("r1"); Resource r2 = new Resource("r2")) {
            r2.use();
        }
        // If both try body AND close() throw, try body exception is primary,
        // close() exception is SUPPRESSED (accessible via getSuppressed())
    }

    static class Resource implements AutoCloseable {
        private final String name;
        Resource(String name) { this.name = name; }
        void use() { System.out.println("Using " + name); }
        @Override public void close() { System.out.println("Closing " + name); }
    }

    // Multi-catch (Java 7+)
    public void multiCatch() {
        try {
            // some code
        } catch (IllegalArgumentException | NullPointerException e) {
            // e is effectively final in multi-catch
            System.out.println(e.getMessage());
        }
    }

    // Exception chaining - preserve original cause
    public void exceptionChaining() throws Exception {
        try {
            throw new SQLException("DB error");
        } catch (Exception e) {
            throw new ServiceException("Service failed", e); // Chain original
        }
    }

    static class ServiceException extends Exception {
        ServiceException(String msg, Throwable cause) { super(msg, cause); }
    }
    static class SQLException extends Exception {
        SQLException(String msg) { super(msg); }
    }

    // TRAP: Catching Error is almost always wrong
    // Error = JVM-level problems (OutOfMemoryError, StackOverflowError)
    // Only catch if you have a specific recovery strategy (rare)

    // StackOverflowError: often from infinite recursion
    public int recursion(int n) {
        return recursion(n - 1); // StackOverflowError (no base case)
    }

    // Custom exception best practices
    static class BusinessException extends RuntimeException {
        private final String errorCode;

        public BusinessException(String errorCode, String message) {
            super(message);
            this.errorCode = errorCode;
        }

        public BusinessException(String errorCode, String message, Throwable cause) {
            super(message, cause);
            this.errorCode = errorCode;
        }

        public String getErrorCode() { return errorCode; }
    }
}

// ============================================================================
// SECTION 8: DESIGN PATTERNS - JAVA IDIOMS
// ============================================================================

/**
 * SINGLETON - Thread-safe implementations
 */
class SingletonPatterns {

    // Option 1: Enum singleton (BEST - handles serialization, reflection attacks)
    enum EnumSingleton {
        INSTANCE;
        public void doSomething() { System.out.println("Singleton action"); }
    }

    // Option 2: Bill Pugh / Initialization-on-demand holder (lazy, thread-safe, no sync overhead)
    static class LazyHolder {
        private LazyHolder() {}

        private static class Holder {
            private static final LazyHolder INSTANCE = new LazyHolder(); // JVM guarantees thread-safe init
        }

        public static LazyHolder getInstance() {
            return Holder.INSTANCE;
        }
    }

    // Option 3: Double-checked locking with volatile (explained in JMM section)
}

/**
 * BUILDER PATTERN - especially for immutable objects
 */
class BuilderDemo {

    static final class HttpRequest {
        private final String url;
        private final String method;
        private final Map<String, String> headers;
        private final String body;
        private final int timeoutMs;

        private HttpRequest(Builder builder) {
            this.url = Objects.requireNonNull(builder.url, "url required");
            this.method = builder.method;
            this.headers = Collections.unmodifiableMap(new HashMap<>(builder.headers));
            this.body = builder.body;
            this.timeoutMs = builder.timeoutMs;
        }

        public static class Builder {
            private String url;
            private String method = "GET";
            private Map<String, String> headers = new HashMap<>();
            private String body;
            private int timeoutMs = 5000;

            public Builder url(String url) { this.url = url; return this; }
            public Builder method(String method) { this.method = method; return this; }
            public Builder header(String key, String value) { headers.put(key, value); return this; }
            public Builder body(String body) { this.body = body; return this; }
            public Builder timeout(int ms) { this.timeoutMs = ms; return this; }
            public HttpRequest build() { return new HttpRequest(this); }
        }
    }

    public void usage() {
        HttpRequest request = new HttpRequest.Builder()
            .url("https://api.example.com")
            .method("POST")
            .header("Content-Type", "application/json")
            .body("{\"key\": \"value\"}")
            .timeout(3000)
            .build();
    }
}

/**
 * OBSERVER PATTERN with Java generics
 */
class ObserverPattern {

    interface EventListener<T> {
        void onEvent(T event);
    }

    static class EventBus<T> {
        private final List<EventListener<T>> listeners = new CopyOnWriteArrayList<>();

        public void subscribe(EventListener<T> listener) { listeners.add(listener); }
        public void unsubscribe(EventListener<T> listener) { listeners.remove(listener); }

        public void publish(T event) {
            listeners.forEach(l -> {
                try { l.onEvent(event); }
                catch (Exception e) { /* Don't let one bad listener break others */ }
            });
        }
    }
}

/**
 * STRATEGY PATTERN with functional interfaces
 */
class StrategyPattern {
    // Java 8+: strategy can be a simple functional interface
    @FunctionalInterface
    interface SortStrategy<T> {
        List<T> sort(List<T> items);
    }

    static class Sorter<T> {
        private SortStrategy<T> strategy;

        Sorter(SortStrategy<T> strategy) { this.strategy = strategy; }
        void setStrategy(SortStrategy<T> strategy) { this.strategy = strategy; }
        List<T> sort(List<T> items) { return strategy.sort(items); }
    }

    public void usage() {
        Sorter<Integer> sorter = new Sorter<>(items -> {
            List<Integer> copy = new ArrayList<>(items);
            Collections.sort(copy);
            return copy;
        });
        // Switch strategy at runtime
        sorter.setStrategy(items -> items.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList()));
    }
}

/**
 * DECORATOR PATTERN
 */
class DecoratorPattern {
    interface DataSource {
        void writeData(String data);
        String readData();
    }

    static class FileDataSource implements DataSource {
        private String filename;
        FileDataSource(String filename) { this.filename = filename; }
        @Override public void writeData(String data) { /* write to file */ }
        @Override public String readData() { return "file data"; }
    }

    static abstract class DataSourceDecorator implements DataSource {
        protected DataSource wrappee;
        DataSourceDecorator(DataSource source) { this.wrappee = source; }
        @Override public void writeData(String data) { wrappee.writeData(data); }
        @Override public String readData() { return wrappee.readData(); }
    }

    static class EncryptionDecorator extends DataSourceDecorator {
        EncryptionDecorator(DataSource source) { super(source); }
        @Override public void writeData(String data) { super.writeData(encrypt(data)); }
        @Override public String readData() { return decrypt(super.readData()); }
        private String encrypt(String data) { return "encrypted:" + data; }
        private String decrypt(String data) { return data.replace("encrypted:", ""); }
    }

    static class CompressionDecorator extends DataSourceDecorator {
        CompressionDecorator(DataSource source) { super(source); }
        @Override public void writeData(String data) { super.writeData(compress(data)); }
        @Override public String readData() { return decompress(super.readData()); }
        private String compress(String data) { return "compressed:" + data; }
        private String decompress(String data) { return data.replace("compressed:", ""); }
    }

    public void usage() {
        DataSource source = new CompressionDecorator(
            new EncryptionDecorator(
                new FileDataSource("data.txt")
            )
        );
        source.writeData("Hello"); // compress(encrypt(writeToFile))
    }
}

// ============================================================================
// SECTION 9: JAVA 8-21 MODERN FEATURES
// ============================================================================

/**
 * RECORDS (Java 14+ preview, 16+ stable)
 * Compact immutable data carriers
 */
record Point(int x, int y) {
    // Compact constructor for validation
    Point {
        if (x < 0 || y < 0) throw new IllegalArgumentException("Coordinates must be non-negative");
    }

    // Can add methods
    public double distanceTo(Point other) {
        int dx = this.x - other.x;
        int dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // equals(), hashCode(), toString() auto-generated
    // Accessors: x() and y() (not getX/getY)
}

/**
 * SEALED CLASSES (Java 17+)
 * Restricted inheritance hierarchy - great for algebraic data types
 */
sealed class Shape permits Circle, Rectangle2, Triangle {
    abstract double area();
}

final class Circle extends Shape {
    private final double radius;
    Circle(double radius) { this.radius = radius; }
    @Override public double area() { return Math.PI * radius * radius; }
}

final class Rectangle2 extends Shape {
    private final double width, height;
    Rectangle2(double w, double h) { this.width = w; this.height = h; }
    @Override public double area() { return width * height; }
}

non-sealed class Triangle extends Shape { // non-sealed: anyone can extend Triangle
    private final double base, height;
    Triangle(double b, double h) { this.base = b; this.height = h; }
    @Override public double area() { return 0.5 * base * height; }
}

/**
 * PATTERN MATCHING (Java 16+)
 * instanceof with binding variable
 */
class PatternMatchingDemo {

    // Old way
    public String describeOld(Object o) {
        if (o instanceof String) {
            String s = (String) o; // explicit cast
            return "String of length " + s.length();
        }
        return "other";
    }

    // New way (Java 16+)
    public String describeNew(Object o) {
        if (o instanceof String s) { // pattern variable 's' automatically cast and bound
            return "String of length " + s.length();
        }
        return "other";
    }

    // Switch expressions (Java 14+) + Pattern matching in switch (Java 21)
    public double areaOf(Shape shape) {
        return switch (shape) {
            case Circle c -> Math.PI * c.area();   // Java 21 pattern matching in switch
            case Rectangle2 r -> r.area();
            case Triangle t -> t.area();
            // No default needed: compiler knows Shape is sealed and all permits covered
        };
    }

    // Switch expression (Java 14+) - no fall-through, returns value
    public String dayType(int day) {
        return switch (day) {
            case 1, 7 -> "Weekend";
            case 2, 3, 4, 5, 6 -> "Weekday";
            default -> throw new IllegalArgumentException("Invalid day: " + day);
        };
    }
}

/**
 * TEXT BLOCKS (Java 15+)
 */
class TextBlocksDemo {
    String json = """
            {
                "name": "John",
                "age": 30
            }
            """; // Trailing newline included

    String html = """
            <html>
                <body>Hello</body>
            </html>
            """.stripIndent(); // stripIndent() removes incidental whitespace
}

/**
 * VAR (Local Variable Type Inference, Java 10+)
 */
class VarDemo {
    public void varUsage() {
        var list = new ArrayList<String>(); // inferred as ArrayList<String>
        var map = new HashMap<String, List<Integer>>(); // saves verbose typing

        // TRAP: var only for LOCAL variables. Cannot use for:
        // - method parameters
        // - return types
        // - fields
        // - catch parameters (Java 10), but allowed in Java 10+

        // TRAP: var doesn't mean dynamic typing. Type is fixed at compile time.
        var x = 42;
        // x = "hello"; // Compile error: x is int

        // TRAP: var with diamond operator loses type info
        // var list2 = new ArrayList<>(); // Inferred as ArrayList<Object>, not helpful
    }
}

// ============================================================================
// SECTION 10: GARBAGE COLLECTION - G1, ZGC, TUNING
// ============================================================================

/**
 * GC ALGORITHMS:
 *
 * Serial GC (-XX:+UseSerialGC): Single-threaded, STW (stop-the-world). Small heaps.
 * Parallel GC (-XX:+UseParallelGC): Multi-threaded STW. Throughput-optimized.
 * CMS (-XX:+UseConcMarkSweepGC): Concurrent mark-sweep, low latency. DEPRECATED.
 * G1 (-XX:+UseG1GC): Default since Java 9. Divides heap into regions.
 * ZGC (-XX:+UseZGC): Java 11+. Sub-millisecond pause times. Scalable.
 * Shenandoah (-XX:+UseShenandoahGC): RedHat. Concurrent compaction.
 *
 * G1 GC REGIONS:
 * - Heap divided into equal-sized regions (1-32MB)
 * - Each region labeled: Eden, Survivor, Old, Humongous (for large objects)
 * - Collects regions with most garbage first (hence "Garbage First")
 * - Mixed GC: collect young + some old regions
 * - Target: meet pause time goal (-XX:MaxGCPauseMillis=200)
 *
 * GC ROOTS (starting points for reachability):
 * - Stack variables (local vars in all thread stacks)
 * - Static fields
 * - JNI references
 * - Class objects loaded by system ClassLoader
 *
 * MEMORY LEAK CAUSES IN JAVA:
 * 1. Static collections holding objects (grow forever)
 * 2. Listeners/callbacks not unregistered
 * 3. ThreadLocal variables not removed (especially in thread pools)
 * 4. Non-static inner class holding reference to outer class
 * 5. Caches without eviction policy
 */
class GCAndReferences {

    // Reference types (affect GC behavior)
    public void referenceTypes() {
        Object obj = new Object();

        // Strong reference: default. GC never collects while reachable.
        Object strong = obj;

        // SoftReference: collected ONLY when JVM needs memory (last resort before OOM)
        SoftReference<Object> soft = new SoftReference<>(obj);
        // Use for: memory-sensitive caches

        // WeakReference: collected at NEXT GC cycle (whenever GC runs)
        WeakReference<Object> weak = new WeakReference<>(obj);
        // Use for: canonicalization maps (WeakHashMap), metadata attached to objects

        // PhantomReference: collected after finalization (for cleanup actions)
        ReferenceQueue<Object> queue = new ReferenceQueue<>();
        PhantomReference<Object> phantom = new PhantomReference<>(obj, queue);
        // get() ALWAYS returns null. Use queue to detect when object is collected.
        // Java 9+: Cleaner API preferred over PhantomReference

        // WeakHashMap: keys are weakly referenced (auto-removed when key GC'd)
        WeakHashMap<Object, String> weakMap = new WeakHashMap<>();
        weakMap.put(obj, "metadata");
        obj = null; // key is now only weakly reachable
        System.gc();
        System.out.println(weakMap.size()); // May be 0 after GC

        // ThreadLocal leak in thread pools
        ThreadLocal<byte[]> threadLocal = new ThreadLocal<>();
        threadLocal.set(new byte[1024 * 1024]); // 1MB per thread
        // If thread is pooled and threadLocal not removed:
        threadLocal.remove(); // ALWAYS call remove() in finally when done
    }

    // GC tuning flags (know these for senior interviews)
    /*
     * -Xms512m                    Initial heap size
     * -Xmx4g                      Max heap size
     * -XX:+UseG1GC                Use G1 GC
     * -XX:MaxGCPauseMillis=200    Target max pause time (G1)
     * -XX:G1HeapRegionSize=16m    G1 region size
     * -XX:NewRatio=3              Old:Young ratio (3:1)
     * -XX:SurvivorRatio=8         Eden:Survivor ratio
     * -XX:+PrintGCDetails         GC logging (use -Xlog:gc* in Java 9+)
     * -XX:+HeapDumpOnOutOfMemoryError  Dump on OOM for analysis
     * -XX:HeapDumpPath=/tmp/heap.hprof
     */
}

// ============================================================================
// SECTION 11: REFLECTION & ANNOTATIONS
// ============================================================================

/**
 * ANNOTATIONS - Custom annotation creation
 */
@Retention(RetentionPolicy.RUNTIME)  // Available at runtime (RUNTIME, CLASS, SOURCE)
@Target({ElementType.METHOD, ElementType.TYPE}) // Where it can be applied
@Documented  // Appear in Javadoc
@interface Retry {
    int maxAttempts() default 3;
    long delayMs() default 1000L;
    Class<? extends Exception>[] on() default {Exception.class};
}

/**
 * REFLECTION - dynamic inspection and invocation
 */
class ReflectionDemo {

    @Retry(maxAttempts = 5, delayMs = 500)
    public String fetchData(String url) {
        return "data";
    }

    public void reflectionUsage() throws Exception {
        Class<?> clazz = ReflectionDemo.class;

        // Inspect methods
        for (Method method : clazz.getDeclaredMethods()) {
            if (method.isAnnotationPresent(Retry.class)) {
                Retry retry = method.getAnnotation(Retry.class);
                System.out.println("Max attempts: " + retry.maxAttempts());
            }
        }

        // Dynamic invocation
        Method method = clazz.getDeclaredMethod("fetchData", String.class);
        method.setAccessible(true); // Bypass access control (careful in production!)
        ReflectionDemo instance = new ReflectionDemo();
        String result = (String) method.invoke(instance, "http://example.com");

        // Get/set private fields
        Field field = clazz.getDeclaredField("someField");
        field.setAccessible(true);
        field.set(instance, "newValue");

        // Create instance dynamically
        Constructor<?> constructor = clazz.getDeclaredConstructor();
        Object obj = constructor.newInstance();

        // TRAP: Reflection bypasses compile-time type checking, slow (30-40x slower than direct call)
        // TRAP: setAccessible(true) may fail with SecurityManager or in Java 9+ module system
    }

    private String someField = "original";

    // Annotation processor (runs at compile time) vs Reflection (runs at runtime)
    // Spring uses BOTH: compile-time (for some things) and runtime reflection for @Autowired etc.
}

// ============================================================================
// SECTION 12: I/O & NIO
// ============================================================================

class IOvsNIO {

    /**
     * TRADITIONAL I/O (java.io):
     * - Blocking: thread blocks until data available
     * - Stream-oriented: reads byte by byte or buffered
     * - Simpler API, but doesn't scale for many connections
     */
    public void traditionalIO() throws IOException {
        // BufferedReader - always buffer for performance
        try (BufferedReader reader = new BufferedReader(
                new FileReader("file.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        }

        // Files.readAllLines (Java 7+) - convenient for small files
        List<String> lines = Files.readAllLines(Path.of("file.txt"));

        // Files.lines (Java 8+) - lazy stream, MUST close stream
        try (Stream<String> stream = Files.lines(Path.of("file.txt"))) {
            stream.filter(l -> l.contains("error")).forEach(System.out::println);
        }
    }

    /**
     * NIO (java.nio):
     * - Non-blocking: single thread can manage multiple connections (Selector)
     * - Buffer-oriented: reads into Buffer objects
     * - Channels: bidirectional (read + write)
     *
     * KEY COMPONENTS: Channel, Buffer, Selector
     */
    public void nioServerSkeleton() throws IOException {
        Selector selector = Selector.open();
        ServerSocketChannel serverChannel = ServerSocketChannel.open();
        serverChannel.bind(new java.net.InetSocketAddress(8080));
        serverChannel.configureBlocking(false); // Non-blocking!
        serverChannel.register(selector, SelectionKey.OP_ACCEPT);

        while (true) {
            selector.select(); // Blocks until at least one channel ready
            Set<SelectionKey> keys = selector.selectedKeys();
            Iterator<SelectionKey> iter = keys.iterator();

            while (iter.hasNext()) {
                SelectionKey key = iter.next();
                iter.remove();

                if (key.isAcceptable()) {
                    SocketChannel client = serverChannel.accept();
                    client.configureBlocking(false);
                    client.register(selector, SelectionKey.OP_READ);
                } else if (key.isReadable()) {
                    SocketChannel client = (SocketChannel) key.channel();
                    ByteBuffer buffer = ByteBuffer.allocate(1024);
                    int bytesRead = client.read(buffer);
                    if (bytesRead == -1) { client.close(); continue; }
                    buffer.flip(); // Switch from write mode to read mode
                    // process buffer data
                }
            }
        }
    }

    /**
     * ByteBuffer modes - COMMON INTERVIEW TRAP
     * After allocate/put: position=written, limit=capacity (WRITE mode)
     * After flip(): position=0, limit=written position (READ mode)
     * After clear(): position=0, limit=capacity (ready to write again)
     * After compact(): unread data moved to start, position after it (read remaining, write more)
     */
    public void byteBufferDemo() {
        ByteBuffer buffer = ByteBuffer.allocate(10); // position=0, limit=10
        buffer.put((byte) 1);                        // position=1
        buffer.put((byte) 2);                        // position=2
        buffer.flip();                               // position=0, limit=2 (READ mode)
        byte b1 = buffer.get();                      // position=1
        byte b2 = buffer.get();                      // position=2, limit=2 -> empty
        buffer.clear();                              // position=0, limit=10 (ready to write)

        // Direct Buffer: allocated outside Java heap (in native memory)
        // Faster for I/O (no copy between Java heap and native), but slower to allocate/GC
        ByteBuffer direct = ByteBuffer.allocateDirect(1024);
    }
}

// ============================================================================
// SECTION 13: SPRING FRAMEWORK INTERNALS (Common in Java Sr. Interviews)
// ============================================================================

/**
 * SPRING IOC CONTAINER:
 * - BeanFactory: basic DI container (lazy by default)
 * - ApplicationContext: extends BeanFactory, adds events, i18n, AOP (eager by default)
 *
 * BEAN LIFECYCLE:
 * 1. Instantiation (constructor)
 * 2. Populate Properties (DI - setter injection or field injection)
 * 3. setBeanName (BeanNameAware)
 * 4. setBeanFactory (BeanFactoryAware)
 * 5. setApplicationContext (ApplicationContextAware)
 * 6. BeanPostProcessor.postProcessBeforeInitialization() (all beans)
 * 7. @PostConstruct / InitializingBean.afterPropertiesSet()
 * 8. Custom init-method
 * 9. BeanPostProcessor.postProcessAfterInitialization() (AOP proxies created here!)
 * 10. READY TO USE
 * 11. @PreDestroy / DisposableBean.destroy()
 * 12. Custom destroy-method
 *
 * SPRING AOP:
 * - JDK dynamic proxy: for beans implementing an interface (proxies interface)
 * - CGLIB proxy: for beans NOT implementing interface (subclasses the class)
 * - TRAP: AOP doesn't work for self-invocation (calling @Transactional method from same class)
 *   because you call through 'this' (not the proxy)
 *
 * TRANSACTION PROPAGATION (know all 7):
 * REQUIRED: Join existing or create new (DEFAULT)
 * REQUIRES_NEW: Always create new, suspend existing
 * SUPPORTS: Join if exists, else non-transactional
 * NOT_SUPPORTED: Suspend existing, run non-transactional
 * MANDATORY: Must join existing, else exception
 * NEVER: Must NOT have transaction, else exception
 * NESTED: Savepoint in existing, or new if none
 *
 * TRANSACTION ISOLATION LEVELS:
 * READ_UNCOMMITTED: Dirty reads possible
 * READ_COMMITTED: No dirty reads, phantom reads possible (PostgreSQL default)
 * REPEATABLE_READ: No dirty/non-repeatable reads, phantom reads possible (MySQL InnoDB default)
 * SERIALIZABLE: No anomalies, lowest performance
 */
class SpringConceptsAsCode {

    // TRAP: @Transactional self-invocation doesn't work
    // @Service
    static class OrderService {
        // @Transactional  // <- This IS transactional (called from outside via proxy)
        public void placeOrder() {
            // process order
            sendEmail(); // <- This call goes through 'this', NOT the proxy!
            // @Transactional on sendEmail() has NO EFFECT for this call
        }

        // @Transactional // <- No effect when called from placeOrder() above
        public void sendEmail() {
            // send email
        }
    }

    // FIX: Inject self, or use @Lazy self-injection, or refactor to separate bean
    // @Service
    static class OrderServiceFixed {
        // @Autowired @Lazy private OrderServiceFixed self; // Inject proxy of self

        public void placeOrder() {
            // self.sendEmail(); // Now goes through proxy, @Transactional works
        }

        // @Transactional
        public void sendEmail() { /* ... */ }
    }

    /**
     * SPRING BEAN SCOPES:
     * singleton: one per ApplicationContext (DEFAULT)
     * prototype: new instance per injection/getBean()
     * request: one per HTTP request (web only)
     * session: one per HTTP session (web only)
     * application: one per ServletContext (web only)
     * websocket: one per WebSocket session
     *
     * TRAP: Injecting prototype bean into singleton - always gets SAME prototype instance
     * FIX: Use ApplicationContext.getBean(), @Lookup, or ObjectFactory<PrototypeBean>
     */
}

// ============================================================================
// SECTION 14: PERFORMANCE, PROFILING & BEST PRACTICES
// ============================================================================

class PerformancePatterns {

    // String concatenation in loops - ALWAYS use StringBuilder
    public String buildString(List<String> parts) {
        // BAD: Creates new String object each iteration O(n^2) total
        // String result = "";
        // for (String p : parts) result += p;

        // GOOD: StringBuilder amortizes allocation
        StringBuilder sb = new StringBuilder(parts.stream().mapToInt(String::length).sum());
        for (String p : parts) sb.append(p);
        return sb.toString();
    }

    // Avoid autoboxing in hot paths
    public long sumList(List<Integer> list) {
        // BAD: unboxes Integer to int on each access
        // long sum = 0;
        // for (Integer i : list) sum += i; // Unboxing

        // GOOD: stream with mapToLong avoids boxing
        return list.stream().mapToLong(Integer::longValue).sum();

        // BETTER: Use primitive array or IntStream to avoid boxing entirely
    }

    // Object pool for expensive objects (JDBC connections, threads)
    // Don't reinvent: use HikariCP for DB pools, ThreadPoolExecutor for thread pools

    // Lazy initialization
    private volatile List<String> expensiveList; // volatile for DCL

    public List<String> getExpensiveList() {
        if (expensiveList == null) {
            synchronized (this) {
                if (expensiveList == null) {
                    expensiveList = computeExpensiveList();
                }
            }
        }
        return expensiveList;
    }

    private List<String> computeExpensiveList() {
        return new ArrayList<>();
    }

    // Efficient collections initialization
    public void collectionInit() {
        // Specify initial capacity to avoid resizing
        List<String> list = new ArrayList<>(1000);
        Map<String, Integer> map = new HashMap<>(1400); // 1000 / 0.75 = 1333, next power of 2

        // Immutable collections (Java 9+) - no copying needed, memory efficient
        List<String> immutable = List.of("a", "b", "c"); // Cannot add/remove
        Map<String, Integer> immutableMap = Map.of("a", 1, "b", 2);
        Set<String> immutableSet = Set.of("x", "y");
        // TRAP: List.of() does NOT allow null elements (throws NPE)
        // Use Collections.unmodifiableList(Arrays.asList(...)) if you need nulls

        // Arrays.asList vs List.of
        List<String> asList = Arrays.asList("a", "b"); // fixed-size, allows set(), no add/remove
        // asList.set(0, "z"); // OK
        // asList.add("c"); // UnsupportedOperationException
    }

    // INTERVIEWER QUESTION: String.format vs MessageFormat vs StringBuilder vs +
    public void stringFormatting() {
        String name = "World";
        int count = 42;

        // + operator: fine for simple cases, compiler uses StringBuilder
        String s1 = "Hello " + name + " count=" + count;

        // String.format: readable but slowest (uses Formatter internally)
        String s2 = String.format("Hello %s count=%d", name, count);

        // StringBuilder: fastest for complex cases
        String s3 = new StringBuilder("Hello ").append(name)
                        .append(" count=").append(count).toString();

        // Java 15+ text blocks + formatted()
        String s4 = "Hello %s count=%d".formatted(name, count);
    }
}

// ============================================================================
// SECTION 15: CRITICAL INTERVIEW TRAPS & GOTCHAS (Quick Reference)
// ============================================================================

class InterviewTraps {

    public void integerCacheTrap() {
        // Integer cache: -128 to 127 are cached (same object)
        Integer a = 127;
        Integer b = 127;
        System.out.println(a == b);   // true (same cached object)

        Integer c = 128;
        Integer d = 128;
        System.out.println(c == d);   // false (different objects)
        System.out.println(c.equals(d)); // true (content equal)
        // LESSON: ALWAYS use .equals() for Integer comparison, NEVER ==
    }

    public void stringImmutabilityTrap() {
        String s = "hello";
        s.toUpperCase(); // Returns new String, original unchanged
        System.out.println(s); // "hello" - unchanged!
        s = s.toUpperCase(); // Must assign
        System.out.println(s); // "HELLO"
    }

    public void arrayVsList() {
        // Arrays are covariant (compile-safe issue)
        Object[] objects = new String[3]; // Legal
        objects[0] = "hello";
        objects[1] = 42; // ArrayStoreException at RUNTIME (not compile time)

        // Generics are invariant (compile-safe)
        // List<Object> objList = new ArrayList<String>(); // COMPILE ERROR (good!)
    }

    public void finallyAndReturn() {
        // finally ALWAYS runs, even with return or exception
        // return in finally overrides return in try
        // Exception in finally masks exception in try
    }

    public void hashCodeContract() {
        // Two objects with same hashCode don't have to be equal
        // Two objects that are equal MUST have same hashCode
        // An object's hashCode should NOT change while in a HashSet/HashMap
    }

    public void concurrentCollections() {
        // Collections.synchronizedList wraps each method with synchronized
        // -> iteration is NOT thread-safe (must sync externally on list object during iteration)
        List<String> syncList = Collections.synchronizedList(new ArrayList<>());
        synchronized (syncList) { // Must lock on the list itself during iteration
            for (String s : syncList) { /* safe */ }
        }

        // CopyOnWriteArrayList: thread-safe for iteration (snapshot), expensive writes
        // ConcurrentHashMap: thread-safe, no locking for reads, segment locking for writes
        // ConcurrentLinkedQueue: lock-free, uses CAS

        // BlockingQueue (ArrayBlockingQueue, LinkedBlockingQueue): producer-consumer
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(100);
        // queue.put("item"); // Blocks if full
        // queue.take();      // Blocks if empty
        // queue.offer("item", 100, TimeUnit.MS); // Wait with timeout
        // queue.poll(100, TimeUnit.MS);           // Wait with timeout
    }

    // TRAP: int division vs long division
    public void arithmeticTraps() {
        System.out.println(5 / 2);            // 2 (integer division, truncates)
        System.out.println(5 / 2.0);          // 2.5
        System.out.println(5.0 / 2);          // 2.5
        System.out.println((double) 5 / 2);   // 2.5

        // Overflow
        int maxInt = Integer.MAX_VALUE;
        System.out.println(maxInt + 1);         // -2147483648 (overflow, no exception)
        System.out.println(Math.addExact(maxInt, 1)); // ArithmeticException (safe)

        // Long literals
        long bigNum = 10_000_000_000L; // L suffix required for long literal > Integer.MAX_VALUE
    }

    // TRAP: varargs and generics
    @SafeVarargs // Suppresses unchecked warning
    public final <T> List<T> asList(T... elements) {
        return Arrays.asList(elements);
    }

    // Heap pollution with varargs generics (why @SafeVarargs exists)
    public void heapPollution() {
        List<String>[] arrayOfLists = new List[2]; // Raw type warning
        Object[] objects = arrayOfLists;
        objects[0] = Arrays.asList(42); // Stores List<Integer> in List<String>[]
        String s = arrayOfLists[0].get(0); // ClassCastException at runtime!
    }
}

// ============================================================================
// SECTION 16: FUNCTIONAL INTERFACES & METHOD REFERENCES
// ============================================================================

class FunctionalInterfacesDemo {

    // Built-in functional interfaces (java.util.function)
    public void builtInFunctionals() {
        // Supplier<T>: () -> T
        Supplier<String> supplier = () -> "hello";
        Supplier<List<String>> listSupplier = ArrayList::new; // Constructor reference

        // Consumer<T>: T -> void
        Consumer<String> printer = System.out::println;       // Instance method ref
        Consumer<String> upper = s -> System.out.println(s.toUpperCase());
        Consumer<String> both = printer.andThen(upper);       // Compose consumers

        // Function<T, R>: T -> R
        Function<String, Integer> length = String::length;    // Instance method ref
        Function<Integer, String> toString = String::valueOf; // Static method ref
        Function<String, String> composed = length.andThen(toString); // Compose
        Function<String, String> composed2 = toString.compose(length); // Reverse compose

        // Predicate<T>: T -> boolean
        Predicate<String> isEmpty = String::isEmpty;
        Predicate<String> isNotEmpty = isEmpty.negate();
        Predicate<String> combined = isEmpty.or(s -> s.equals("N/A"));
        Predicate<String> both2 = isNotEmpty.and(s -> s.length() > 3);

        // BiFunction<T, U, R>: (T, U) -> R
        BiFunction<String, Integer, String> repeat = (s, n) -> s.repeat(n);

        // UnaryOperator<T>: T -> T (special Function where input=output type)
        UnaryOperator<String> trim = String::trim;

        // BinaryOperator<T>: (T, T) -> T
        BinaryOperator<Integer> sum = Integer::sum;
        BinaryOperator<Integer> max = BinaryOperator.maxBy(Comparator.naturalOrder());
    }

    // Method reference types
    public void methodReferences() {
        // 1. Static method: Class::staticMethod
        Function<String, Integer> parseInt = Integer::parseInt;

        // 2. Instance method on specific instance: instance::method
        String prefix = "Hello";
        Predicate<String> startsWith = prefix::startsWith; // specific instance 'prefix'

        // 3. Instance method on arbitrary instance: Class::instanceMethod
        Function<String, String> trim = String::trim; // called on the argument

        // 4. Constructor: Class::new
        Supplier<ArrayList<String>> listFactory = ArrayList::new;
        Function<Integer, ArrayList<String>> sized = ArrayList::new; // with capacity
    }

    // Closure - lambda captures effectively final variables
    public void closureTrap() {
        int x = 10; // effectively final (not modified after assignment)
        Runnable r = () -> System.out.println(x); // captures x

        // x = 20; // COMPILE ERROR: would make x NOT effectively final
        // Runnable r2 = () -> System.out.println(x); // then this fails

        // TRAP: Captured variables must be effectively final
        // Workaround for mutable capture:
        int[] counter = {0}; // Array is effectively final (reference), but content mutable
        Runnable incrementer = () -> counter[0]++;
        incrementer.run();
        System.out.println(counter[0]); // 1
        // This works but is fragile for concurrent use - prefer AtomicInteger
    }
}

// ============================================================================
// SECTION 17: COMMON ALGORITHMIC PATTERNS (Java-specific implementation)
// ============================================================================

class AlgorithmicPatterns {

    // Custom Comparable for complex sorting
    static class Task implements Comparable<Task> {
        int priority;
        String name;
        LocalDateTime deadline;

        Task(int priority, String name, LocalDateTime deadline) {
            this.priority = priority;
            this.name = name;
            this.deadline = deadline;
        }

        @Override
        public int compareTo(Task other) {
            // First by priority (higher = more urgent)
            int cmp = Integer.compare(other.priority, this.priority);
            if (cmp != 0) return cmp;
            // Then by deadline (earlier = more urgent)
            return this.deadline.compareTo(other.deadline);
        }
    }

    // PriorityQueue for scheduling
    public void schedulingDemo() {
        PriorityQueue<Task> taskQueue = new PriorityQueue<>();
        // Poll returns lowest (min-heap by default, need to customize for max)
        PriorityQueue<Task> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());

        // Top-K elements
        PriorityQueue<Integer> topK = new PriorityQueue<>(5); // min-heap size k
        int[] data = {3,1,4,1,5,9,2,6,5,3};
        for (int n : data) {
            topK.offer(n);
            if (topK.size() > 5) topK.poll(); // remove smallest, keep top 5
        }
    }

    // Sliding Window
    public int maxSumSubarray(int[] arr, int k) {
        int windowSum = 0;
        for (int i = 0; i < k; i++) windowSum += arr[i];
        int maxSum = windowSum;
        for (int i = k; i < arr.length; i++) {
            windowSum += arr[i] - arr[i - k];
            maxSum = Math.max(maxSum, windowSum);
        }
        return maxSum;
    }

    // Two pointers
    public boolean hasPairWithSum(int[] sortedArr, int target) {
        int left = 0, right = sortedArr.length - 1;
        while (left < right) {
            int sum = sortedArr[left] + sortedArr[right];
            if (sum == target) return true;
            else if (sum < target) left++;
            else right--;
        }
        return false;
    }

    // Binary search - always verify edge cases
    public int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2; // NEVER (left+right)/2 -> overflow risk
            if (arr[mid] == target) return mid;
            else if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

    // LRU Cache using LinkedHashMap
    static class LRUCache<K, V> extends LinkedHashMap<K, V> {
        private final int capacity;

        LRUCache(int capacity) {
            super(capacity, 0.75f, true); // accessOrder=true: MRU order
            this.capacity = capacity;
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
            return size() > capacity;
        }
    }
}

/*
============================================================================
QUICK REFERENCE: KEY DIFFERENCES
============================================================================

== vs equals():
  - ==: reference equality (same object in memory)
  - equals(): logical equality (content, defined by the class)
  - ALWAYS use equals() for objects, == only for primitives or intentional ref check

StringBuilder vs StringBuffer:
  - StringBuilder: NOT thread-safe, faster (use in single-threaded)
  - StringBuffer: thread-safe (synchronized methods), slower

ArrayList vs LinkedList:
  - ArrayList: O(1) get by index, O(n) insert/delete middle, contiguous memory (cache-friendly)
  - LinkedList: O(n) get by index, O(1) insert/delete at known node, pointer overhead
  - In practice: ArrayList almost always better (cache locality wins)

HashMap vs Hashtable:
  - HashMap: NOT thread-safe, allows null key/values
  - Hashtable: thread-safe (synchronized), no null keys/values, LEGACY (use ConcurrentHashMap)

Vector vs ArrayList:
  - Vector: synchronized, LEGACY (use ArrayList + explicit sync or CopyOnWriteArrayList)

Fail-fast vs Fail-safe iterators:
  - Fail-fast: ArrayList, HashMap - throws CME if modified during iteration
  - Fail-safe: CopyOnWriteArrayList, ConcurrentHashMap - iterates over snapshot

Abstract Class vs Interface:
  - Abstract: can have state (instance fields), constructors, protected methods
  - Interface: no state (only static final), no constructors, all public
  - Use abstract class when: sharing code between related classes
  - Use interface when: defining a contract, multiple inheritance of type

throw vs throws:
  - throw: actually throws an exception (in code)
  - throws: declares that method may throw a checked exception (in signature)

final vs finally vs finalize:
  - final: keyword for variable (constant ref), method (no override), class (no extend)
  - finally: block that always runs after try/catch
  - finalize(): deprecated Object method called by GC before collection (avoid, use Cleaner)
============================================================================
*/
