// ============================================================================
// .NET INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
// For 8+ Years SDE Experience
// ============================================================================

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Reflection;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Net.Http;

// ============================================================================
// CORE C# FUNDAMENTALS
// ============================================================================

#region Value Types vs Reference Types

public class ValueVsReferenceTypes
{
    // Value types: stored on stack, copied by value
    // int, double, bool, struct, enum, DateTime, Guid
    public void ValueTypeExample()
    {
        int a = 10;
        int b = a;  // Copy by value
        b = 20;
        Console.WriteLine(a);  // 10 (unchanged)
    }

    // Reference types: stored on heap, copied by reference
    // class, interface, delegate, object, string, arrays
    public void ReferenceTypeExample()
    {
        var list1 = new List<int> { 1, 2, 3 };
        var list2 = list1;  // Copy by reference
        list2.Add(4);
        Console.WriteLine(list1.Count);  // 4 (changed)
    }

    // Struct (value type)
    public struct Point
    {
        public int X { get; set; }
        public int Y { get; set; }

        public Point(int x, int y)
        {
            X = x;
            Y = y;
        }
    }

    // Class (reference type)
    public class Rectangle
    {
        public int Width { get; set; }
        public int Height { get; set; }

        public Rectangle(int width, int height)
        {
            Width = width;
            Height = height;
        }
    }

    // Boxing and Unboxing
    public void BoxingUnboxing()
    {
        int i = 123;
        object o = i;        // Boxing (value type → reference type)
        int j = (int)o;      // Unboxing (reference type → value type)

        // Performance cost of boxing
        List<object> list = new List<object>();
        for (int x = 0; x < 1000; x++)
        {
            list.Add(x);  // Boxing on each iteration (avoid!)
        }

        // Better: use generic list
        List<int> betterList = new List<int>();
        for (int x = 0; x < 1000; x++)
        {
            betterList.Add(x);  // No boxing
        }
    }
}

#endregion

#region Memory Management and Garbage Collection

public class MemoryManagement
{
    // Garbage Collection Generations
    // Gen 0: Short-lived objects (newly allocated)
    // Gen 1: Transition generation
    // Gen 2: Long-lived objects

    public void GCExample()
    {
        // Force garbage collection (avoid in production)
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // Check generation of an object
        var obj = new object();
        int generation = GC.GetGeneration(obj);
        Console.WriteLine($"Object is in generation: {generation}");

        // Get total memory
        long memory = GC.GetTotalMemory(false);
        Console.WriteLine($"Total memory: {memory} bytes");
    }

    // IDisposable pattern for unmanaged resources
    public class ResourceHolder : IDisposable
    {
        private IntPtr unmanagedResource;
        private bool disposed = false;

        public ResourceHolder()
        {
            // Allocate unmanaged resource
        }

        // Public Dispose method
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // Protected Dispose method
        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources
                }

                // Dispose unmanaged resources
                if (unmanagedResource != IntPtr.Zero)
                {
                    // Free unmanaged resource
                    unmanagedResource = IntPtr.Zero;
                }

                disposed = true;
            }
        }

        // Finalizer (only if you have unmanaged resources)
        ~ResourceHolder()
        {
            Dispose(false);
        }
    }

    // Using statement (automatic disposal)
    public void UsingStatementExample()
    {
        using (var resource = new ResourceHolder())
        {
            // Use resource
        } // Dispose called automatically

        // C# 8.0+ using declaration
        using var resource2 = new ResourceHolder();
        // Dispose called at end of scope
    }

    // Weak References
    public void WeakReferenceExample()
    {
        var strongRef = new byte[1024 * 1024];  // 1MB
        var weakRef = new WeakReference(strongRef);

        Console.WriteLine(weakRef.IsAlive);  // True

        strongRef = null;
        GC.Collect();

        Console.WriteLine(weakRef.IsAlive);  // False (may be collected)
    }

    // Memory Leaks - Common Causes
    public class MemoryLeakExamples
    {
        // Leak 1: Event handlers not unsubscribed
        public event EventHandler MyEvent;
        private List<byte[]> data = new List<byte[]>();

        public void Subscribe()
        {
            MyEvent += OnMyEvent;
        }

        public void Unsubscribe()
        {
            MyEvent -= OnMyEvent;  // Always unsubscribe!
        }

        private void OnMyEvent(object sender, EventArgs e) { }

        // Leak 2: Static references
        private static List<object> staticList = new List<object>();

        public void AddToStaticList(object obj)
        {
            staticList.Add(obj);  // Never garbage collected!
        }

        // Leak 3: Timer not disposed
        private Timer timer;

        public void StartTimer()
        {
            timer = new Timer(callback: _ => Console.WriteLine("Tick"), 
                            state: null, 
                            dueTime: 0, 
                            period: 1000);
        }

        public void StopTimer()
        {
            timer?.Dispose();  // Must dispose!
        }
    }
}

#endregion

#region Delegates, Events, and Lambda Expressions

public class DelegatesAndEvents
{
    // Delegate declaration
    public delegate void NotifyDelegate(string message);
    public delegate int CalculateDelegate(int x, int y);

    public void DelegateBasics()
    {
        // Single method
        NotifyDelegate notify = ShowMessage;
        notify("Hello");

        // Multicast delegate
        notify += ShowMessageUpperCase;
        notify("World");  // Both methods called

        // Remove from multicast
        notify -= ShowMessage;
        notify("Test");  // Only ShowMessageUpperCase called
    }

    private void ShowMessage(string message)
    {
        Console.WriteLine(message);
    }

    private void ShowMessageUpperCase(string message)
    {
        Console.WriteLine(message.ToUpper());
    }

    // Built-in delegates
    public void BuiltInDelegates()
    {
        // Action: void return, up to 16 parameters
        Action<string> action = (msg) => Console.WriteLine(msg);
        action("Hello");

        // Func: non-void return, up to 16 parameters
        Func<int, int, int> add = (a, b) => a + b;
        int result = add(5, 3);

        // Predicate: bool return, single parameter
        Predicate<int> isEven = (n) => n % 2 == 0;
        bool even = isEven(4);
    }

    // Events
    public class Publisher
    {
        // Event declaration
        public event EventHandler<DataEventArgs> DataReceived;

        public void RaiseEvent(string data)
        {
            // Null-conditional operator for thread safety
            DataReceived?.Invoke(this, new DataEventArgs(data));
        }
    }

    public class DataEventArgs : EventArgs
    {
        public string Data { get; }
        public DataEventArgs(string data) => Data = data;
    }

    public class Subscriber
    {
        public void Subscribe(Publisher publisher)
        {
            publisher.DataReceived += OnDataReceived;
        }

        public void Unsubscribe(Publisher publisher)
        {
            publisher.DataReceived -= OnDataReceived;
        }

        private void OnDataReceived(object sender, DataEventArgs e)
        {
            Console.WriteLine($"Received: {e.Data}");
        }
    }

    // Lambda expressions and closures
    public void LambdaExamples()
    {
        // Expression lambda
        Func<int, int> square = x => x * x;

        // Statement lambda
        Action<string> print = (msg) =>
        {
            Console.WriteLine($"Message: {msg}");
        };

        // Closure: capturing outer variable
        int factor = 5;
        Func<int, int> multiply = x => x * factor;
        Console.WriteLine(multiply(3));  // 15

        factor = 10;
        Console.WriteLine(multiply(3));  // 30 (captures by reference)
    }

    // Expression trees
    public void ExpressionTreeExample()
    {
        // Lambda compiled to expression tree
        System.Linq.Expressions.Expression<Func<int, bool>> expr = 
            x => x > 5;

        // Can analyze the expression
        Console.WriteLine(expr.Body);  // (x > 5)
        
        // Compile and execute
        var compiled = expr.Compile();
        bool result = compiled(10);  // true
    }
}

#endregion

#region LINQ and Collections

public class LinqExamples
{
    public void QuerySyntaxVsMethodSyntax()
    {
        var numbers = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        // Query syntax
        var queryResult = from n in numbers
                         where n % 2 == 0
                         select n * 2;

        // Method syntax (preferred for complex queries)
        var methodResult = numbers
            .Where(n => n % 2 == 0)
            .Select(n => n * 2);

        // Both are equivalent: [4, 8, 12, 16, 20]
    }

    public void CommonLinqOperations()
    {
        var numbers = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        // Filtering
        var evens = numbers.Where(n => n % 2 == 0);

        // Projection
        var squares = numbers.Select(n => n * n);

        // Ordering
        var descending = numbers.OrderByDescending(n => n);

        // Aggregation
        int sum = numbers.Sum();
        int max = numbers.Max();
        double avg = numbers.Average();
        int count = numbers.Count(n => n > 5);

        // First, Last, Single
        int first = numbers.First();  // Throws if empty
        int firstOrDefault = numbers.FirstOrDefault();  // Returns default if empty
        int single = numbers.Single(n => n == 5);  // Throws if 0 or >1 matches

        // Any, All
        bool hasEvens = numbers.Any(n => n % 2 == 0);
        bool allPositive = numbers.All(n => n > 0);

        // GroupBy
        var grouped = numbers.GroupBy(n => n % 3);
        foreach (var group in grouped)
        {
            Console.WriteLine($"Key: {group.Key}");
            foreach (var item in group)
            {
                Console.WriteLine($"  {item}");
            }
        }

        // Join
        var persons = new[]
        {
            new { Id = 1, Name = "Alice" },
            new { Id = 2, Name = "Bob" }
        };

        var orders = new[]
        {
            new { PersonId = 1, Product = "Book" },
            new { PersonId = 2, Product = "Pen" }
        };

        var joined = from p in persons
                     join o in orders on p.Id equals o.PersonId
                     select new { p.Name, o.Product };
    }

    public void DeferredVsImmediateExecution()
    {
        var numbers = new List<int> { 1, 2, 3 };

        // Deferred execution (query not executed yet)
        var query = numbers.Where(n => n > 1);

        numbers.Add(4);

        // Query executes now - includes 4!
        var result = query.ToList();  // [2, 3, 4]

        // Immediate execution
        var immediateResult = numbers.Where(n => n > 1).ToList();
        numbers.Add(5);
        // immediateResult still [2, 3, 4] - doesn't include 5
    }

    // Custom LINQ operator
    public static class CustomLinqExtensions
    {
        public static IEnumerable<T> WhereNot<T>(
            this IEnumerable<T> source, 
            Func<T, bool> predicate)
        {
            foreach (var item in source)
            {
                if (!predicate(item))
                    yield return item;
            }
        }
    }

    // Parallel LINQ (PLINQ)
    public void ParallelLinqExample()
    {
        var numbers = Enumerable.Range(1, 1000000);

        // Sequential
        var sequentialSum = numbers.Sum();

        // Parallel
        var parallelSum = numbers.AsParallel().Sum();

        // With ordering preserved
        var orderedResult = numbers
            .AsParallel()
            .AsOrdered()
            .Where(n => n % 2 == 0)
            .Take(10);

        // Degree of parallelism
        var customParallel = numbers
            .AsParallel()
            .WithDegreeOfParallelism(4)
            .Sum();
    }
}

public class CollectionTypes
{
    public void ListExample()
    {
        var list = new List<int> { 1, 2, 3 };
        list.Add(4);
        list.AddRange(new[] { 5, 6 });
        list.Insert(0, 0);
        list.Remove(3);
        list.RemoveAt(0);
        bool contains = list.Contains(5);
        int index = list.IndexOf(2);

        // List is O(1) for access by index
        int value = list[2];
    }

    public void DictionaryExample()
    {
        var dict = new Dictionary<string, int>
        {
            ["one"] = 1,
            ["two"] = 2
        };

        dict.Add("three", 3);
        dict["four"] = 4;  // Add or update

        bool hasKey = dict.ContainsKey("one");
        bool hasValue = dict.ContainsValue(1);

        if (dict.TryGetValue("one", out int value))
        {
            Console.WriteLine(value);
        }

        // Iterate
        foreach (var kvp in dict)
        {
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");
        }
    }

    public void HashSetExample()
    {
        var set = new HashSet<int> { 1, 2, 3, 3 };  // Duplicates ignored
        set.Add(4);
        bool added = set.Add(1);  // false (already exists)

        // Set operations
        var set2 = new HashSet<int> { 3, 4, 5 };
        set.UnionWith(set2);        // [1, 2, 3, 4, 5]
        set.IntersectWith(set2);    // [3, 4, 5]
        set.ExceptWith(set2);       // [1, 2]
    }

    public void QueueAndStackExample()
    {
        // Queue: FIFO
        var queue = new Queue<int>();
        queue.Enqueue(1);
        queue.Enqueue(2);
        int first = queue.Dequeue();  // 1
        int peek = queue.Peek();      // 2 (doesn't remove)

        // Stack: LIFO
        var stack = new Stack<int>();
        stack.Push(1);
        stack.Push(2);
        int top = stack.Pop();  // 2
        int peekTop = stack.Peek();  // 1
    }

    public void ConcurrentCollectionsExample()
    {
        // Thread-safe collections
        var concurrentDict = new ConcurrentDictionary<string, int>();
        concurrentDict.TryAdd("key", 1);
        concurrentDict.AddOrUpdate("key", 1, (k, v) => v + 1);

        var concurrentQueue = new ConcurrentQueue<int>();
        concurrentQueue.Enqueue(1);
        if (concurrentQueue.TryDequeue(out int result))
        {
            Console.WriteLine(result);
        }

        var concurrentBag = new ConcurrentBag<int>();
        concurrentBag.Add(1);
        if (concurrentBag.TryTake(out int item))
        {
            Console.WriteLine(item);
        }
    }

    // Custom collection
    public class CustomCollection<T> : IEnumerable<T>
    {
        private List<T> items = new List<T>();

        public void Add(T item) => items.Add(item);

        public IEnumerator<T> GetEnumerator()
        {
            foreach (var item in items)
            {
                yield return item;
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}

#endregion

#region Async/Await and Task Parallel Library

public class AsyncProgramming
{
    // Basic async/await
    public async Task<string> FetchDataAsync()
    {
        await Task.Delay(1000);  // Simulate async operation
        return "Data";
    }

    // Async void (only for event handlers)
    public async void Button_Click(object sender, EventArgs e)
    {
        await FetchDataAsync();
        // Can't be awaited, exceptions are harder to handle
    }

    // Multiple async operations
    public async Task<string> ParallelAsyncOperations()
    {
        // Sequential (slower)
        var result1 = await FetchDataAsync();
        var result2 = await FetchDataAsync();

        // Parallel (faster)
        var task1 = FetchDataAsync();
        var task2 = FetchDataAsync();
        await Task.WhenAll(task1, task2);
        var parallelResult1 = task1.Result;
        var parallelResult2 = task2.Result;

        return parallelResult1 + parallelResult2;
    }

    // Task.WhenAll vs Task.WhenAny
    public async Task TaskCombinators()
    {
        var tasks = new[]
        {
            Task.Delay(1000),
            Task.Delay(2000),
            Task.Delay(3000)
        };

        // Wait for all
        await Task.WhenAll(tasks);

        // Wait for first
        var completedTask = await Task.WhenAny(tasks);

        // Timeout pattern
        var dataTask = FetchDataAsync();
        var timeoutTask = Task.Delay(5000);
        var completed = await Task.WhenAny(dataTask, timeoutTask);

        if (completed == timeoutTask)
        {
            throw new TimeoutException("Operation timed out");
        }

        return await dataTask;
    }

    // CancellationToken
    public async Task<string> CancellableOperation(CancellationToken ct)
    {
        for (int i = 0; i < 10; i++)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Delay(1000, ct);
        }
        return "Completed";
    }

    public async Task UseCancellation()
    {
        var cts = new CancellationTokenSource();
        cts.CancelAfter(5000);  // Cancel after 5 seconds

        try
        {
            await CancellableOperation(cts.Token);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("Operation cancelled");
        }
    }

    // ConfigureAwait
    public async Task ConfigureAwaitExample()
    {
        // Captures synchronization context (UI thread, ASP.NET context)
        await Task.Delay(1000);

        // Doesn't capture context (better performance for library code)
        await Task.Delay(1000).ConfigureAwait(false);
    }

    // ValueTask (when result is often synchronous)
    public ValueTask<int> GetCachedOrFetch(string key)
    {
        if (cache.TryGetValue(key, out int value))
        {
            return new ValueTask<int>(value);  // Synchronous
        }

        return new ValueTask<int>(FetchFromDbAsync(key));  // Asynchronous
    }

    private Dictionary<string, int> cache = new Dictionary<string, int>();
    private async Task<int> FetchFromDbAsync(string key)
    {
        await Task.Delay(100);
        return 42;
    }

    // Task Parallel Library (TPL)
    public void ParallelForExample()
    {
        // Parallel.For
        Parallel.For(0, 100, i =>
        {
            Console.WriteLine($"Processing {i} on thread {Thread.CurrentThread.ManagedThreadId}");
        });

        // Parallel.ForEach
        var items = Enumerable.Range(1, 100);
        Parallel.ForEach(items, item =>
        {
            // Process item in parallel
        });

        // Parallel.Invoke
        Parallel.Invoke(
            () => Method1(),
            () => Method2(),
            () => Method3()
        );
    }

    private void Method1() { }
    private void Method2() { }
    private void Method3() { }

    // Async enumerable (C# 8.0+)
    public async IAsyncEnumerable<int> GenerateNumbersAsync()
    {
        for (int i = 0; i < 10; i++)
        {
            await Task.Delay(100);
            yield return i;
        }
    }

    public async Task ConsumeAsyncEnumerable()
    {
        await foreach (var number in GenerateNumbersAsync())
        {
            Console.WriteLine(number);
        }
    }

    // SemaphoreSlim for async synchronization
    private SemaphoreSlim semaphore = new SemaphoreSlim(3);  // Max 3 concurrent

    public async Task AccessResourceAsync()
    {
        await semaphore.WaitAsync();
        try
        {
            // Access limited resource
            await Task.Delay(1000);
        }
        finally
        {
            semaphore.Release();
        }
    }
}

#endregion

#region Thread Safety and Synchronization

public class ThreadSafety
{
    private int counter = 0;
    private object lockObject = new object();

    // Lock statement
    public void IncrementWithLock()
    {
        lock (lockObject)
        {
            counter++;
        }
    }

    // Interlocked for atomic operations
    public void IncrementWithInterlocked()
    {
        Interlocked.Increment(ref counter);
        Interlocked.Add(ref counter, 5);
        int original = Interlocked.Exchange(ref counter, 100);
        int previous = Interlocked.CompareExchange(ref counter, 200, 100);
    }

    // Monitor (more control than lock)
    public void MonitorExample()
    {
        if (Monitor.TryEnter(lockObject, TimeSpan.FromSeconds(5)))
        {
            try
            {
                counter++;
            }
            finally
            {
                Monitor.Exit(lockObject);
            }
        }
    }

    // ReaderWriterLockSlim
    private ReaderWriterLockSlim rwLock = new ReaderWriterLockSlim();
    private Dictionary<string, int> data = new Dictionary<string, int>();

    public int ReadData(string key)
    {
        rwLock.EnterReadLock();
        try
        {
            return data[key];
        }
        finally
        {
            rwLock.ExitReadLock();
        }
    }

    public void WriteData(string key, int value)
    {
        rwLock.EnterWriteLock();
        try
        {
            data[key] = value;
        }
        finally
        {
            rwLock.ExitWriteLock();
        }
    }

    // Mutex (cross-process synchronization)
    public void MutexExample()
    {
        using (var mutex = new Mutex(false, "Global\\MyAppMutex"))
        {
            if (mutex.WaitOne(TimeSpan.FromSeconds(5)))
            {
                try
                {
                    // Critical section
                }
                finally
                {
                    mutex.ReleaseMutex();
                }
            }
        }
    }

    // Semaphore
    private Semaphore semaphore = new Semaphore(3, 3);  // Max 3 threads

    public void SemaphoreExample()
    {
        semaphore.WaitOne();
        try
        {
            // Access limited resource
        }
        finally
        {
            semaphore.Release();
        }
    }

    // Thread-local storage
    private ThreadLocal<int> threadLocalValue = new ThreadLocal<int>(() => 0);

    public void ThreadLocalExample()
    {
        threadLocalValue.Value = Thread.CurrentThread.ManagedThreadId;
        Console.WriteLine(threadLocalValue.Value);  // Different per thread
    }

    // Volatile keyword
    private volatile bool stopRequested = false;

    public void VolatileExample()
    {
        // Ensures reads/writes are not reordered by compiler/CPU
        stopRequested = true;
    }

    // Lazy initialization (thread-safe)
    private Lazy<ExpensiveObject> lazyObject = 
        new Lazy<ExpensiveObject>(() => new ExpensiveObject());

    public ExpensiveObject GetExpensiveObject()
    {
        return lazyObject.Value;  // Created only once, thread-safe
    }

    private class ExpensiveObject { }
}

#endregion

#region Exception Handling

public class ExceptionHandling
{
    // Basic try-catch-finally
    public void BasicExceptionHandling()
    {
        try
        {
            int result = 10 / 0;
        }
        catch (DivideByZeroException ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Unexpected error: {ex.Message}");
            throw;  // Re-throw preserves stack trace
        }
        finally
        {
            // Always executes (cleanup code)
        }
    }

    // Exception filters (C# 6.0+)
    public void ExceptionFilters()
    {
        try
        {
            // Some operation
        }
        catch (Exception ex) when (ex.Message.Contains("timeout"))
        {
            // Handle timeout specifically
        }
        catch (Exception ex) when (LogException(ex))
        {
            // This catch never executes (LogException returns false)
            // But logging happens during filter evaluation
        }
    }

    private bool LogException(Exception ex)
    {
        Console.WriteLine($"Logged: {ex.Message}");
        return false;  // Don't catch
    }

    // Custom exceptions
    public class ValidationException : Exception
    {
        public string PropertyName { get; }

        public ValidationException(string propertyName, string message)
            : base(message)
        {
            PropertyName = propertyName;
        }

        public ValidationException(string propertyName, string message, Exception inner)
            : base(message, inner)
        {
            PropertyName = propertyName;
        }
    }

    // Exception handling in async code
    public async Task AsyncExceptionHandling()
    {
        try
        {
            await Task.Run(() => throw new InvalidOperationException());
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine(ex.Message);
        }

        // Multiple tasks
        var tasks = new[]
        {
            Task.Run(() => throw new Exception("Error 1")),
            Task.Run(() => throw new Exception("Error 2"))
        };

        try
        {
            await Task.WhenAll(tasks);
        }
        catch (Exception ex)
        {
            // Only first exception caught
            Console.WriteLine(ex.Message);
        }

        // To catch all exceptions
        try
        {
            await Task.WhenAll(tasks);
        }
        catch
        {
            foreach (var task in tasks)
            {
                if (task.Exception != null)
                {
                    foreach (var inner in task.Exception.InnerExceptions)
                    {
                        Console.WriteLine(inner.Message);
                    }
                }
            }
        }
    }

    // AggregateException
    public void AggregateExceptionHandling()
    {
        try
        {
            var tasks = new[]
            {
                Task.Run(() => throw new InvalidOperationException()),
                Task.Run(() => throw new ArgumentException())
            };

            Task.WaitAll(tasks);
        }
        catch (AggregateException ae)
        {
            ae.Handle(ex =>
            {
                if (ex is InvalidOperationException)
                {
                    Console.WriteLine("InvalidOperation handled");
                    return true;  // Handled
                }
                return false;  // Not handled, will be re-thrown
            });
        }
    }
}

#endregion

#region Dependency Injection and IoC

public interface ILogger
{
    void Log(string message);
}

public class ConsoleLogger : ILogger
{
    public void Log(string message)
    {
        Console.WriteLine($"[LOG] {message}");
    }
}

public interface IRepository<T>
{
    Task<T> GetByIdAsync(int id);
    Task<IEnumerable<T>> GetAllAsync();
    Task AddAsync(T entity);
}

public class UserRepository : IRepository<User>
{
    private readonly ILogger logger;

    // Constructor injection
    public UserRepository(ILogger logger)
    {
        this.logger = logger;
    }

    public async Task<User> GetByIdAsync(int id)
    {
        logger.Log($"Getting user {id}");
        await Task.Delay(100);
        return new User { Id = id, Name = "Test User" };
    }

    public async Task<IEnumerable<User>> GetAllAsync()
    {
        await Task.Delay(100);
        return new List<User>();
    }

    public async Task AddAsync(User entity)
    {
        await Task.Delay(100);
    }
}

public class User
{
    public int Id { get; set; }
    public string Name { get; set; }
}

public class UserService
{
    private readonly IRepository<User> repository;
    private readonly ILogger logger;

    // Multiple dependencies
    public UserService(IRepository<User> repository, ILogger logger)
    {
        this.repository = repository;
        this.logger = logger;
    }

    public async Task<User> GetUserAsync(int id)
    {
        logger.Log($"UserService: Getting user {id}");
        return await repository.GetByIdAsync(id);
    }
}

// Service lifetimes understanding
// Transient: New instance every time
// Scoped: One instance per request/scope
// Singleton: One instance for entire application lifetime

public class DependencyInjectionSetup
{
    // Simulated DI container setup (like in ASP.NET Core)
    public void ConfigureServices(IServiceCollection services)
    {
        // Transient
        services.AddTransient<ILogger, ConsoleLogger>();

        // Scoped
        services.AddScoped<IRepository<User>, UserRepository>();

        // Singleton
        services.AddSingleton<ICacheService, MemoryCacheService>();

        // Generic registration
        services.AddScoped(typeof(IRepository<>), typeof(Repository<>));
    }
}

public interface ICacheService
{
    void Set(string key, object value);
    object Get(string key);
}

public class MemoryCacheService : ICacheService
{
    private readonly Dictionary<string, object> cache = new Dictionary<string, object>();

    public void Set(string key, object value) => cache[key] = value;
    public object Get(string key) => cache.TryGetValue(key, out var value) ? value : null;
}

public class Repository<T> : IRepository<T>
{
    public Task<T> GetByIdAsync(int id) => throw new NotImplementedException();
    public Task<IEnumerable<T>> GetAllAsync() => throw new NotImplementedException();
    public Task AddAsync(T entity) => throw new NotImplementedException();
}

public interface IServiceCollection
{
    void AddTransient<TInterface, TImplementation>();
    void AddScoped<TInterface, TImplementation>();
    void AddSingleton<TInterface, TImplementation>();
    void AddScoped(Type serviceType, Type implementationType);
}

#endregion

#region Generics and Constraints

public class GenericsExamples
{
    // Generic class
    public class Repository<T> where T : class
    {
        private List<T> items = new List<T>();

        public void Add(T item) => items.Add(item);
        public T Get(int index) => items[index];
        public IEnumerable<T> GetAll() => items;
    }

    // Generic method
    public T GetDefault<T>()
    {
        return default(T);  // null for reference types, 0 for value types
    }

    // Multiple type parameters
    public class KeyValuePair<TKey, TValue>
    {
        public TKey Key { get; set; }
        public TValue Value { get; set; }
    }

    // Generic constraints
    public class ConstrainedGeneric<T> where T : class, IDisposable, new()
    {
        // T must be:
        // - Reference type (class)
        // - Implement IDisposable
        // - Have parameterless constructor (new())

        public T CreateAndUse()
        {
            T instance = new T();
            try
            {
                return instance;
            }
            finally
            {
                instance.Dispose();
            }
        }
    }

    // Covariance (out) - can return more derived types
    public interface IReadOnlyRepository<out T>
    {
        T Get(int id);
        IEnumerable<T> GetAll();
        // Can't have T as input parameter with 'out'
    }

    // Contravariance (in) - can accept more derived types
    public interface IComparer<in T>
    {
        int Compare(T x, T y);
        // Can't return T with 'in'
    }

    public void CovarianceExample()
    {
        IReadOnlyRepository<string> stringRepo = null;
        IReadOnlyRepository<object> objectRepo = stringRepo;  // Covariance
    }

    public void ContravarianceExample()
    {
        IComparer<object> objectComparer = null;
        IComparer<string> stringComparer = objectComparer;  // Contravariance
    }

    // Generic constraints examples
    public T Max<T>(T a, T b) where T : IComparable<T>
    {
        return a.CompareTo(b) > 0 ? a : b;
    }

    public void Process<T>(T item) where T : struct
    {
        // T must be value type
    }

    public void Clone<T>(T source, T destination) 
        where T : class, ICloneable
    {
        // T must be reference type and implement ICloneable
    }

    // Generic method with multiple constraints
    public TResult Convert<TSource, TResult>(TSource source)
        where TSource : class
        where TResult : class, new()
    {
        return new TResult();
    }
}

#endregion

#region Reflection and Attributes

public class ReflectionExamples
{
    // Getting type information
    public void TypeInformation()
    {
        Type stringType = typeof(string);
        Type intType = typeof(int);

        string str = "hello";
        Type strInstanceType = str.GetType();

        // Type properties
        Console.WriteLine(stringType.Name);           // String
        Console.WriteLine(stringType.FullName);       // System.String
        Console.WriteLine(stringType.Namespace);      // System
        Console.WriteLine(stringType.IsClass);        // True
        Console.WriteLine(stringType.IsValueType);    // False
    }

    // Activator - creating instances
    public void CreateInstanceExample()
    {
        // Create instance of type
        var list = (List<int>)Activator.CreateInstance(typeof(List<int>));

        // With parameters
        var dateTime = (DateTime)Activator.CreateInstance(
            typeof(DateTime), 2024, 1, 1);

        // Generic version
        var genericList = Activator.CreateInstance<List<string>>();
    }

    // Property reflection
    public void PropertyReflection()
    {
        var person = new Person { Name = "Alice", Age = 30 };
        Type type = typeof(Person);

        // Get all properties
        PropertyInfo[] properties = type.GetProperties();

        foreach (var prop in properties)
        {
            Console.WriteLine($"{prop.Name}: {prop.GetValue(person)}");
        }

        // Get specific property
        PropertyInfo nameProp = type.GetProperty("Name");
        nameProp.SetValue(person, "Bob");

        // Get property value
        object nameValue = nameProp.GetValue(person);
    }

    // Method reflection
    public void MethodReflection()
    {
        Type type = typeof(Calculator);
        
        // Get method
        MethodInfo addMethod = type.GetMethod("Add");

        // Invoke method
        var calculator = new Calculator();
        object result = addMethod.Invoke(calculator, new object[] { 5, 3 });
        Console.WriteLine(result);  // 8

        // Get all methods
        MethodInfo[] methods = type.GetMethods(
            BindingFlags.Public | 
            BindingFlags.Instance | 
            BindingFlags.DeclaredOnly);
    }

    // Generic method invocation
    public void GenericMethodReflection()
    {
        Type type = typeof(GenericClass);
        MethodInfo method = type.GetMethod("GenericMethod");
        MethodInfo generic = method.MakeGenericMethod(typeof(int));

        var instance = Activator.CreateInstance(type);
        generic.Invoke(instance, new object[] { 42 });
    }

    // Custom attributes
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
    public class AuthorAttribute : Attribute
    {
        public string Name { get; }
        public string Date { get; set; }

        public AuthorAttribute(string name)
        {
            Name = name;
        }
    }

    [Author("Alice", Date = "2024-01-01")]
    public class DocumentedClass
    {
        [Author("Bob")]
        public void DocumentedMethod() { }
    }

    // Reading attributes
    public void ReadAttributes()
    {
        Type type = typeof(DocumentedClass);

        // Get class attributes
        var classAttrs = type.GetCustomAttributes<AuthorAttribute>();
        foreach (var attr in classAttrs)
        {
            Console.WriteLine($"Class author: {attr.Name}, Date: {attr.Date}");
        }

        // Get method attributes
        MethodInfo method = type.GetMethod("DocumentedMethod");
        var methodAttr = method.GetCustomAttribute<AuthorAttribute>();
        Console.WriteLine($"Method author: {methodAttr.Name}");
    }

    // Assembly loading
    public void AssemblyReflection()
    {
        // Current assembly
        Assembly currentAssembly = Assembly.GetExecutingAssembly();

        // Load assembly
        Assembly assembly = Assembly.Load("System.Text.Json");

        // Get all types from assembly
        Type[] types = assembly.GetTypes();

        // Get specific type
        Type jsonType = assembly.GetType("System.Text.Json.JsonSerializer");
    }
}

public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }
}

public class Calculator
{
    public int Add(int a, int b) => a + b;
}

public class GenericClass
{
    public void GenericMethod<T>(T value)
    {
        Console.WriteLine($"Value: {value}, Type: {typeof(T)}");
    }
}

#endregion

#region Expression Trees and Dynamic

public class ExpressionTreeExamples
{
    public void BasicExpressionTree()
    {
        // Lambda expression
        Func<int, int> square = x => x * x;

        // Expression tree
        System.Linq.Expressions.Expression<Func<int, int>> exprSquare = x => x * x;

        // Compile and execute
        var compiled = exprSquare.Compile();
        int result = compiled(5);  // 25
    }

    public void BuildExpressionTree()
    {
        // Build: x => x * x
        var parameter = System.Linq.Expressions.Expression.Parameter(typeof(int), "x");
        var multiply = System.Linq.Expressions.Expression.Multiply(parameter, parameter);
        var lambda = System.Linq.Expressions.Expression.Lambda<Func<int, int>>(multiply, parameter);

        var compiled = lambda.Compile();
        Console.WriteLine(compiled(5));  // 25
    }

    public void DynamicExample()
    {
        // Dynamic type - resolved at runtime
        dynamic dyn = "Hello";
        Console.WriteLine(dyn.Length);  // 5

        dyn = 42;
        Console.WriteLine(dyn + 10);  // 52

        dyn = new System.Dynamic.ExpandoObject();
        dyn.Name = "Alice";
        dyn.Age = 30;
        Console.WriteLine($"{dyn.Name}, {dyn.Age}");
    }

    // Dynamic object
    public class DynamicDictionary : System.Dynamic.DynamicObject
    {
        private Dictionary<string, object> dictionary = new Dictionary<string, object>();

        public override bool TryGetMember(System.Dynamic.GetMemberBinder binder, out object result)
        {
            return dictionary.TryGetValue(binder.Name, out result);
        }

        public override bool TrySetMember(System.Dynamic.SetMemberBinder binder, object value)
        {
            dictionary[binder.Name] = value;
            return true;
        }
    }

    public void UseDynamicDictionary()
    {
        dynamic obj = new DynamicDictionary();
        obj.Name = "Alice";
        obj.Age = 30;
        Console.WriteLine($"{obj.Name}, {obj.Age}");
    }
}

#endregion

#region Serialization

public class SerializationExamples
{
    // JSON serialization (System.Text.Json)
    public void JsonSerialization()
    {
        var person = new PersonDto 
        { 
            Name = "Alice", 
            Age = 30,
            Email = "alice@example.com"
        };

        // Serialize
        string json = System.Text.Json.JsonSerializer.Serialize(person);
        Console.WriteLine(json);

        // Deserialize
        var deserializedPerson = System.Text.Json.JsonSerializer.Deserialize<PersonDto>(json);

        // Custom options
        var options = new System.Text.Json.JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };

        string formattedJson = System.Text.Json.JsonSerializer.Serialize(person, options);
    }

    // Custom JSON converter
    public class DateOnlyConverter : System.Text.Json.Serialization.JsonConverter<DateTime>
    {
        public override DateTime Read(
            ref System.Text.Json.Utf8JsonReader reader, 
            Type typeToConvert, 
            System.Text.Json.JsonSerializerOptions options)
        {
            return DateTime.Parse(reader.GetString());
        }

        public override void Write(
            System.Text.Json.Utf8JsonWriter writer, 
            DateTime value, 
            System.Text.Json.JsonSerializerOptions options)
        {
            writer.WriteStringValue(value.ToString("yyyy-MM-dd"));
        }
    }

    // XML serialization
    public void XmlSerialization()
    {
        var person = new PersonDto { Name = "Alice", Age = 30 };

        var serializer = new System.Xml.Serialization.XmlSerializer(typeof(PersonDto));

        // Serialize
        using (var writer = new StringWriter())
        {
            serializer.Serialize(writer, person);
            string xml = writer.ToString();
            Console.WriteLine(xml);
        }

        // Deserialize
        using (var reader = new StringReader("<xml>"))
        {
            var deserializedPerson = (PersonDto)serializer.Deserialize(reader);
        }
    }

    // Binary serialization (legacy, avoid in new code)
    // Prefer protobuf, MessagePack, or JSON for new applications
}

[System.Text.Json.Serialization.JsonConverter(typeof(System.Text.Json.Serialization.JsonStringEnumConverter))]
public enum Status
{
    Active,
    Inactive
}

public class PersonDto
{
    public string Name { get; set; }
    public int Age { get; set; }

    [System.Text.Json.Serialization.JsonIgnore]
    public string Password { get; set; }

    [System.Text.Json.Serialization.JsonPropertyName("email_address")]
    public string Email { get; set; }

    public Status Status { get; set; }
}

#endregion

#region Design Patterns

public class DesignPatterns
{
    // Singleton pattern
    public sealed class Singleton
    {
        private static readonly Lazy<Singleton> lazy = 
            new Lazy<Singleton>(() => new Singleton());

        public static Singleton Instance => lazy.Value;

        private Singleton() { }

        public void DoSomething()
        {
            Console.WriteLine("Singleton method");
        }
    }

    // Factory pattern
    public interface IProduct
    {
        string GetName();
    }

    public class ProductA : IProduct
    {
        public string GetName() => "Product A";
    }

    public class ProductB : IProduct
    {
        public string GetName() => "Product B";
    }

    public class ProductFactory
    {
        public IProduct CreateProduct(string type)
        {
            return type switch
            {
                "A" => new ProductA(),
                "B" => new ProductB(),
                _ => throw new ArgumentException("Invalid type")
            };
        }
    }

    // Abstract Factory pattern
    public interface IUIFactory
    {
        IButton CreateButton();
        ITextBox CreateTextBox();
    }

    public interface IButton { }
    public interface ITextBox { }

    public class WindowsFactory : IUIFactory
    {
        public IButton CreateButton() => new WindowsButton();
        public ITextBox CreateTextBox() => new WindowsTextBox();
    }

    public class MacFactory : IUIFactory
    {
        public IButton CreateButton() => new MacButton();
        public ITextBox CreateTextBox() => new MacTextBox();
    }

    public class WindowsButton : IButton { }
    public class WindowsTextBox : ITextBox { }
    public class MacButton : IButton { }
    public class MacTextBox : ITextBox { }

    // Builder pattern
    public class Product
    {
        public string PartA { get; set; }
        public string PartB { get; set; }
        public string PartC { get; set; }
    }

    public class ProductBuilder
    {
        private Product product = new Product();

        public ProductBuilder SetPartA(string partA)
        {
            product.PartA = partA;
            return this;
        }

        public ProductBuilder SetPartB(string partB)
        {
            product.PartB = partB;
            return this;
        }

        public ProductBuilder SetPartC(string partC)
        {
            product.PartC = partC;
            return this;
        }

        public Product Build() => product;
    }

    public void UseBuilder()
    {
        var product = new ProductBuilder()
            .SetPartA("A")
            .SetPartB("B")
            .SetPartC("C")
            .Build();
    }

    // Observer pattern
    public interface IObserver
    {
        void Update(string message);
    }

    public class Subject
    {
        private List<IObserver> observers = new List<IObserver>();

        public void Attach(IObserver observer)
        {
            observers.Add(observer);
        }

        public void Detach(IObserver observer)
        {
            observers.Remove(observer);
        }

        public void Notify(string message)
        {
            foreach (var observer in observers)
            {
                observer.Update(message);
            }
        }
    }

    public class ConcreteObserver : IObserver
    {
        private string name;

        public ConcreteObserver(string name)
        {
            this.name = name;
        }

        public void Update(string message)
        {
            Console.WriteLine($"{name} received: {message}");
        }
    }

    // Strategy pattern
    public interface IPaymentStrategy
    {
        void Pay(decimal amount);
    }

    public class CreditCardPayment : IPaymentStrategy
    {
        public void Pay(decimal amount)
        {
            Console.WriteLine($"Paid {amount} with credit card");
        }
    }

    public class PayPalPayment : IPaymentStrategy
    {
        public void Pay(decimal amount)
        {
            Console.WriteLine($"Paid {amount} with PayPal");
        }
    }

    public class ShoppingCart
    {
        private IPaymentStrategy paymentStrategy;

        public void SetPaymentStrategy(IPaymentStrategy strategy)
        {
            paymentStrategy = strategy;
        }

        public void Checkout(decimal amount)
        {
            paymentStrategy.Pay(amount);
        }
    }

    // Decorator pattern
    public interface IComponent
    {
        string Operation();
    }

    public class ConcreteComponent : IComponent
    {
        public string Operation() => "ConcreteComponent";
    }

    public abstract class Decorator : IComponent
    {
        protected IComponent component;

        public Decorator(IComponent component)
        {
            this.component = component;
        }

        public virtual string Operation()
        {
            return component.Operation();
        }
    }

    public class ConcreteDecoratorA : Decorator
    {
        public ConcreteDecoratorA(IComponent component) : base(component) { }

        public override string Operation()
        {
            return $"DecoratorA({base.Operation()})";
        }
    }

    public class ConcreteDecoratorB : Decorator
    {
        public ConcreteDecoratorB(IComponent component) : base(component) { }

        public override string Operation()
        {
            return $"DecoratorB({base.Operation()})";
        }
    }

    // Repository pattern
    public interface IGenericRepository<T> where T : class
    {
        Task<T> GetByIdAsync(int id);
        Task<IEnumerable<T>> GetAllAsync();
        Task AddAsync(T entity);
        Task UpdateAsync(T entity);
        Task DeleteAsync(int id);
    }

    public class GenericRepository<T> : IGenericRepository<T> where T : class
    {
        // DbContext would be injected here
        public Task<T> GetByIdAsync(int id) => throw new NotImplementedException();
        public Task<IEnumerable<T>> GetAllAsync() => throw new NotImplementedException();
        public Task AddAsync(T entity) => throw new NotImplementedException();
        public Task UpdateAsync(T entity) => throw new NotImplementedException();
        public Task DeleteAsync(int id) => throw new NotImplementedException();
    }

    // Unit of Work pattern
    public interface IUnitOfWork : IDisposable
    {
        IGenericRepository<User> Users { get; }
        IGenericRepository<Order> Orders { get; }
        Task<int> SaveChangesAsync();
    }

    public class Order { }
}

#endregion

#region Performance Optimization

public class PerformanceOptimization
{
    // String concatenation
    public void StringPerformance()
    {
        // Bad - creates new string each time
        string result = "";
        for (int i = 0; i < 1000; i++)
        {
            result += i.ToString();  // Avoid!
        }

        // Good - StringBuilder
        var sb = new StringBuilder();
        for (int i = 0; i < 1000; i++)
        {
            sb.Append(i);
        }
        string betterResult = sb.ToString();

        // String interpolation vs String.Format vs concatenation
        string name = "Alice";
        int age = 30;

        string s1 = $"Name: {name}, Age: {age}";  // String interpolation (fastest)
        string s2 = String.Format("Name: {0}, Age: {1}", name, age);
        string s3 = "Name: " + name + ", Age: " + age;  // Concatenation
    }

    // Span<T> and Memory<T> for high-performance scenarios
    public void SpanExample()
    {
        string text = "Hello World";
        
        // Span - stack-allocated, no heap allocation
        ReadOnlySpan<char> span = text.AsSpan();
        ReadOnlySpan<char> hello = span.Slice(0, 5);

        // No string allocation
        Console.WriteLine(hello.ToString());
    }

    public void MemoryExample()
    {
        byte[] buffer = new byte[1024];
        Memory<byte> memory = buffer.AsMemory();
        Memory<byte> slice = memory.Slice(0, 512);
    }

    // ArrayPool for reducing allocations
    public void ArrayPoolExample()
    {
        var pool = System.Buffers.ArrayPool<byte>.Shared;

        byte[] buffer = pool.Rent(1024);  // Rent from pool
        try
        {
            // Use buffer
        }
        finally
        {
            pool.Return(buffer);  // Return to pool
        }
    }

    // ValueTask for high-performance async
    public async ValueTask<int> GetValueAsync(int id)
    {
        if (cache.TryGetValue(id, out int value))
        {
            return value;  // Synchronous path, no Task allocation
        }

        // Asynchronous path
        value = await FetchFromDatabaseAsync(id);
        cache[id] = value;
        return value;
    }

    private Dictionary<int, int> cache = new Dictionary<int, int>();
    private async Task<int> FetchFromDatabaseAsync(int id)
    {
        await Task.Delay(100);
        return id * 10;
    }

    // Struct vs Class performance
    public struct SmallStruct  // Good for small, immutable data
    {
        public int X { get; }
        public int Y { get; }

        public SmallStruct(int x, int y)
        {
            X = x;
            Y = y;
        }
    }

    public readonly struct ReadOnlyStruct  // C# 7.2+, enforces immutability
    {
        public int Value { get; }
        public ReadOnlyStruct(int value) => Value = value;
    }

    // ref struct (stack-only, can't be boxed)
    public ref struct RefStruct
    {
        public Span<int> Data { get; }
        public RefStruct(Span<int> data) => Data = data;
    }

    // in, ref, out parameters
    public void ParameterOptimization()
    {
        var largeStruct = new LargeStruct();

        // Bad - copies entire struct
        ProcessValue(largeStruct);

        // Good - passes by reference (no copy)
        ProcessByRef(in largeStruct);
    }

    private void ProcessValue(LargeStruct value) { }
    private void ProcessByRef(in LargeStruct value) { }  // 'in' = read-only reference

    public struct LargeStruct
    {
        public long A, B, C, D, E, F, G, H;
    }

    // Avoid LINQ in hot paths
    public void LinqPerformance()
    {
        var numbers = Enumerable.Range(1, 1000).ToArray();

        // Slower - LINQ
        var linqSum = numbers.Where(n => n % 2 == 0).Sum();

        // Faster - for loop
        int forSum = 0;
        for (int i = 0; i < numbers.Length; i++)
        {
            if (numbers[i] % 2 == 0)
                forSum += numbers[i];
        }
    }
}

#endregion

#region Entity Framework Core Patterns

public class EFCorePatterns
{
    // DbContext example
    public class ApplicationDbContext //: DbContext
    {
        // public DbSet<User> Users { get; set; }
        // public DbSet<Order> Orders { get; set; }

        // protected override void OnModelCreating(ModelBuilder modelBuilder)
        // {
        //     // Fluent API configuration
        //     modelBuilder.Entity<User>(entity =>
        //     {
        //         entity.HasKey(e => e.Id);
        //         entity.Property(e => e.Name).IsRequired().HasMaxLength(100);
        //         entity.HasIndex(e => e.Email).IsUnique();
        //     });

        //     modelBuilder.Entity<Order>(entity =>
        //     {
        //         entity.HasKey(e => e.Id);
        //         entity.HasOne(e => e.User)
        //               .WithMany(u => u.Orders)
        //               .HasForeignKey(e => e.UserId);
        //     });
        // }
    }

    // Tracking vs No-Tracking queries
    public async Task QueryPatterns(ApplicationDbContext context)
    {
        // Tracking query (default) - EF tracks changes
        // var user = await context.Users.FirstOrDefaultAsync(u => u.Id == 1);
        // user.Name = "Updated";
        // await context.SaveChangesAsync();  // EF detects and saves changes

        // No-tracking query (read-only, better performance)
        // var users = await context.Users.AsNoTracking().ToListAsync();
    }

    // Eager loading, explicit loading, lazy loading
    public async Task LoadingPatterns(ApplicationDbContext context)
    {
        // Eager loading (Include)
        // var user = await context.Users
        //     .Include(u => u.Orders)
        //     .ThenInclude(o => o.OrderItems)
        //     .FirstOrDefaultAsync(u => u.Id == 1);

        // Explicit loading
        // var user = await context.Users.FindAsync(1);
        // await context.Entry(user).Collection(u => u.Orders).LoadAsync();

        // Lazy loading (requires proxies, generally avoid)
        // var user = await context.Users.FindAsync(1);
        // var orders = user.Orders;  // Lazy loaded
    }

    // Projection (select only needed columns)
    public async Task ProjectionPattern(ApplicationDbContext context)
    {
        // Bad - loads entire entity
        // var users = await context.Users.ToListAsync();

        // Good - projects to DTO
        // var userDtos = await context.Users
        //     .Select(u => new UserDto
        //     {
        //         Name = u.Name,
        //         Email = u.Email
        //     })
        //     .ToListAsync();
    }

    // Batch operations
    public async Task BatchOperations(ApplicationDbContext context)
    {
        // AddRange for multiple inserts
        // var users = new List<User>
        // {
        //     new User { Name = "Alice" },
        //     new User { Name = "Bob" }
        // };
        // context.Users.AddRange(users);
        // await context.SaveChangesAsync();

        // ExecuteUpdateAsync (EF Core 7+)
        // await context.Users
        //     .Where(u => u.IsActive)
        //     .ExecuteUpdateAsync(s => s.SetProperty(u => u.LastUpdated, DateTime.UtcNow));

        // ExecuteDeleteAsync (EF Core 7+)
        // await context.Users
        //     .Where(u => u.IsDeleted)
        //     .ExecuteDeleteAsync();
    }

    // Compiled queries for better performance
    // private static readonly Func<ApplicationDbContext, int, Task<User>> GetUserById =
    //     EF.CompileAsyncQuery((ApplicationDbContext context, int id) =>
    //         context.Users.FirstOrDefault(u => u.Id == id));

    public async Task CompiledQueryExample(ApplicationDbContext context)
    {
        // var user = await GetUserById(context, 1);
    }
}

#endregion

#region ASP.NET Core Concepts

public class AspNetCorePatterns
{
    // Middleware pipeline
    public class CustomMiddleware
    {
        private readonly RequestDelegate next;

        public CustomMiddleware(RequestDelegate next)
        {
            this.next = next;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            // Before next middleware
            Console.WriteLine("Before");

            await next(context);  // Call next middleware

            // After next middleware
            Console.WriteLine("After");
        }
    }

    // Action filters
    public class LoggingActionFilter // : IActionFilter
    {
        public void OnActionExecuting(object context)
        {
            Console.WriteLine("Action executing");
        }

        public void OnActionExecuted(object context)
        {
            Console.WriteLine("Action executed");
        }
    }

    // Model validation
    public class UserCreateModel
    {
        // [Required]
        // [StringLength(100)]
        public string Name { get; set; }

        // [Required]
        // [EmailAddress]
        public string Email { get; set; }

        // [Range(18, 100)]
        public int Age { get; set; }
    }

    // Custom validation attribute
    public class FutureDateAttribute : ValidationAttribute
    {
        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            if (value is DateTime date)
            {
                if (date > DateTime.Now)
                {
                    return ValidationResult.Success;
                }
                return new ValidationResult("Date must be in the future");
            }
            return new ValidationResult("Invalid date");
        }
    }

    // API versioning approach
    // [ApiController]
    // [Route("api/v{version:apiVersion}/[controller]")]
    // [ApiVersion("1.0")]
    // public class UsersController : ControllerBase
    // {
    //     [HttpGet]
    //     public IActionResult GetUsers() => Ok();
    // }

    // Response caching
    // [ResponseCache(Duration = 60, Location = ResponseCacheLocation.Client)]
    public string GetCachedData()
    {
        return "Cached data";
    }

    // CORS configuration concept
    public void ConfigureCors()
    {
        // services.AddCors(options =>
        // {
        //     options.AddPolicy("AllowSpecificOrigin",
        //         builder => builder
        //             .WithOrigins("https://example.com")
        //             .AllowAnyMethod()
        //             .AllowAnyHeader());
        // });
    }
}

#endregion

#region Advanced C# Features

public class AdvancedCSharpFeatures
{
    // Pattern matching (C# 7.0+)
    public void PatternMatching()
    {
        object obj = 42;

        // Type pattern
        if (obj is int number)
        {
            Console.WriteLine($"Integer: {number}");
        }

        // Switch expression (C# 8.0)
        string result = obj switch
        {
            int i => $"Integer: {i}",
            string s => $"String: {s}",
            null => "Null",
            _ => "Unknown"
        };

        // Property pattern
        var point = new Point3D { X = 1, Y = 2, Z = 3 };
        string classification = point switch
        {
            { X: 0, Y: 0, Z: 0 } => "Origin",
            { Z: 0 } => "On XY plane",
            _ => "3D point"
        };

        // Relational and logical patterns (C# 9.0)
        int age = 25;
        string category = age switch
        {
            < 18 => "Minor",
            >= 18 and < 65 => "Adult",
            >= 65 => "Senior",
            _ => "Unknown"
        };
    }

    public class Point3D
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Z { get; set; }
    }

    // Record types (C# 9.0) - immutable reference types
    public record PersonRecord(string Name, int Age);

    public void RecordExample()
    {
        var person1 = new PersonRecord("Alice", 30);
        var person2 = new PersonRecord("Alice", 30);

        Console.WriteLine(person1 == person2);  // True (value equality)

        // With expression (non-destructive mutation)
        var person3 = person1 with { Age = 31 };
    }

    // Init-only properties (C# 9.0)
    public class PersonWithInit
    {
        public string Name { get; init; }
        public int Age { get; init; }
    }

    public void InitOnlyExample()
    {
        var person = new PersonWithInit { Name = "Alice", Age = 30 };
        // person.Name = "Bob";  // Compile error - init only
    }

    // Target-typed new (C# 9.0)
    public void TargetTypedNew()
    {
        PersonWithInit person = new() { Name = "Alice", Age = 30 };
        List<int> numbers = new() { 1, 2, 3 };
    }

    // Required members (C# 11.0)
    public class PersonRequired
    {
        public required string Name { get; init; }
        public int Age { get; init; }
    }

    public void RequiredMemberExample()
    {
        // var person = new PersonRequired { Age = 30 };  // Compile error
        var person = new PersonRequired { Name = "Alice", Age = 30 };  // OK
    }

    // Raw string literals (C# 11.0)
    public void RawStringLiterals()
    {
        string json = """
            {
                "name": "Alice",
                "age": 30
            }
            """;

        // With interpolation
        string name = "Alice";
        string interpolated = $"""
            {
                "name": "{{name}}"
            }
            """;
    }

    // List patterns (C# 11.0)
    public void ListPatterns()
    {
        int[] numbers = { 1, 2, 3, 4 };

        string result = numbers switch
        {
            [] => "Empty",
            [var first] => $"Single: {first}",
            [var first, var second] => $"Two: {first}, {second}",
            [1, 2, ..] => "Starts with 1, 2",
            [.., 9, 10] => "Ends with 9, 10",
            _ => "Other"
        };
    }

    // Local functions
    public int CalculateFactorial(int n)
    {
        return Factorial(n);

        int Factorial(int num)
        {
            if (num <= 1) return 1;
            return num * Factorial(num - 1);
        }
    }

    // Tuple deconstruction
    public void TupleDeconstruction()
    {
        var tuple = (Name: "Alice", Age: 30);

        // Deconstruction
        var (name, age) = tuple;
        Console.WriteLine($"{name}, {age}");

        // Discard unwanted values
        var (_, ageOnly) = tuple;
    }

    public (int Sum, int Count) GetStats(int[] numbers)
    {
        return (numbers.Sum(), numbers.Length);
    }

    // Nullable reference types (C# 8.0)
    public void NullableReferences()
    {
        string? nullableString = null;  // Can be null
        string nonNullableString = "Hello";  // Cannot be null

        // Null-forgiving operator
        string value = nullableString!;  // Suppress warning

        // Null-conditional operator
        int? length = nullableString?.Length;

        // Null-coalescing operator
        string result = nullableString ?? "default";

        // Null-coalescing assignment
        nullableString ??= "assigned";
    }

    // Index and Range (C# 8.0)
    public void IndexAndRange()
    {
        int[] numbers = { 1, 2, 3, 4, 5 };

        int last = numbers[^1];      // Last element (5)
        int secondLast = numbers[^2]; // Second to last (4)

        int[] slice = numbers[1..4];  // [2, 3, 4]
        int[] fromStart = numbers[..3]; // [1, 2, 3]
        int[] toEnd = numbers[2..];   // [3, 4, 5]
        int[] all = numbers[..];      // All elements
    }

    // Static local functions
    public void StaticLocalFunction()
    {
        int x = 10;

        // Regular local function can capture x
        int AddX(int y) => x + y;

        // Static local function cannot capture x (better performance)
        static int Add(int a, int b) => a + b;
    }

    // Caller information attributes
    public void LogMethod(
        string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        Console.WriteLine($"{memberName} at {filePath}:{lineNumber} - {message}");
    }
}

#endregion

#region Security Best Practices

public class SecurityPatterns
{
    // Password hashing
    public string HashPassword(string password)
    {
        // Use BCrypt or ASP.NET Core Identity's password hasher
        // Never store plain text passwords!
        
        // Example with BCrypt (requires BCrypt.Net-Next package)
        // return BCrypt.Net.BCrypt.HashPassword(password);
        
        return "hashed";
    }

    public bool VerifyPassword(string password, string hash)
    {
        // return BCrypt.Net.BCrypt.Verify(password, hash);
        return true;
    }

    // Encryption/Decryption
    public byte[] EncryptData(byte[] data, byte[] key, byte[] iv)
    {
        using (var aes = System.Security.Cryptography.Aes.Create())
        {
            aes.Key = key;
            aes.IV = iv;

            using (var encryptor = aes.CreateEncryptor())
            using (var ms = new MemoryStream())
            {
                using (var cs = new System.Security.Cryptography.CryptoStream(
                    ms, encryptor, System.Security.Cryptography.CryptoStreamMode.Write))
                {
                    cs.Write(data, 0, data.Length);
                }
                return ms.ToArray();
            }
        }
    }

    // SQL Injection prevention
    public async Task<User> GetUserSafe(int userId)
    {
        // Good - parameterized query
        // using (var connection = new SqlConnection(connectionString))
        // {
        //     var command = new SqlCommand(
        //         "SELECT * FROM Users WHERE Id = @UserId", 
        //         connection);
        //     command.Parameters.AddWithValue("@UserId", userId);
        //     // Execute query
        // }

        // Or use EF Core / Dapper (automatically parameterized)
        return null;
    }

    // XSS prevention
    public string SanitizeHtml(string input)
    {
        // Use HtmlEncoder
        return System.Text.Encodings.Web.HtmlEncoder.Default.Encode(input);
    }

    // CSRF token validation (handled by ASP.NET Core)
    // [ValidateAntiForgeryToken]
    public void PostAction()
    {
        // Token automatically validated
    }

    // Secure random number generation
    public byte[] GenerateSecureRandomBytes(int length)
    {
        using (var rng = System.Security.Cryptography.RandomNumberGenerator.Create())
        {
            byte[] randomBytes = new byte[length];
            rng.GetBytes(randomBytes);
            return randomBytes;
        }
    }

    // JWT token handling concept
    public string GenerateJwtToken(User user, string secretKey)
    {
        // Using System.IdentityModel.Tokens.Jwt
        // var tokenHandler = new JwtSecurityTokenHandler();
        // var key = Encoding.ASCII.GetBytes(secretKey);
        // var tokenDescriptor = new SecurityTokenDescriptor
        // {
        //     Subject = new ClaimsIdentity(new[]
        //     {
        //         new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
        //         new Claim(ClaimTypes.Name, user.Name)
        //     }),
        //     Expires = DateTime.UtcNow.AddHours(1),
        //     SigningCredentials = new SigningCredentials(
        //         new SymmetricSecurityKey(key),
        //         SecurityAlgorithms.HmacSha256Signature)
        // };
        // var token = tokenHandler.CreateToken(tokenDescriptor);
        // return tokenHandler.WriteToken(token);
        
        return "token";
    }
}

#endregion

#region Testing Patterns

public class TestingPatterns
{
    // Unit testing example (using xUnit, NUnit, or MSTest)
    public class Calculator
    {
        public int Add(int a, int b) => a + b;
        public int Divide(int a, int b)
        {
            if (b == 0) throw new DivideByZeroException();
            return a / b;
        }
    }

    // Example test class
    public class CalculatorTests
    {
        // [Fact] // xUnit
        public void Add_TwoPositiveNumbers_ReturnsSum()
        {
            // Arrange
            var calculator = new Calculator();

            // Act
            int result = calculator.Add(2, 3);

            // Assert
            // Assert.Equal(5, result);
        }

        // [Theory] // xUnit
        // [InlineData(10, 2, 5)]
        // [InlineData(15, 3, 5)]
        public void Divide_ValidNumbers_ReturnsQuotient(int a, int b, int expected)
        {
            var calculator = new Calculator();
            int result = calculator.Divide(a, b);
            // Assert.Equal(expected, result);
        }

        // [Fact]
        public void Divide_ByZero_ThrowsException()
        {
            var calculator = new Calculator();
            // Assert.Throws<DivideByZeroException>(() => calculator.Divide(10, 0));
        }
    }

    // Mocking with Moq
    public class UserServiceTests
    {
        // [Fact]
        public async Task GetUser_ExistingId_ReturnsUser()
        {
            // Arrange
            // var mockRepo = new Mock<IRepository<User>>();
            // var mockLogger = new Mock<ILogger>();
            
            // var expectedUser = new User { Id = 1, Name = "Alice" };
            // mockRepo.Setup(r => r.GetByIdAsync(1))
            //     .ReturnsAsync(expectedUser);

            // var service = new UserService(mockRepo.Object, mockLogger.Object);

            // Act
            // var result = await service.GetUserAsync(1);

            // Assert
            // Assert.Equal(expectedUser.Name, result.Name);
            // mockLogger.Verify(l => l.Log(It.IsAny<string>()), Times.Once);
        }
    }

    // Integration testing
    public class IntegrationTests
    {
        // [Fact]
        public async Task GetUsers_ReturnsOk()
        {
            // Using WebApplicationFactory for ASP.NET Core
            // var factory = new WebApplicationFactory<Program>();
            // var client = factory.CreateClient();
            
            // var response = await client.GetAsync("/api/users");
            
            // response.EnsureSuccessStatusCode();
            // var content = await response.Content.ReadAsStringAsync();
        }
    }

    // Test fixtures and shared context
    public class DatabaseFixture : IDisposable
    {
        public ApplicationDbContext Context { get; private set; }

        public DatabaseFixture()
        {
            // Setup database
        }

        public void Dispose()
        {
            // Cleanup
        }
    }

    public class DatabaseTests // : IClassFixture<DatabaseFixture>
    {
        private readonly DatabaseFixture fixture;

        public DatabaseTests(DatabaseFixture fixture)
        {
            this.fixture = fixture;
        }
    }
}

#endregion

#region Common Algorithms in C#

public class CommonAlgorithms
{
    // Binary search
    public int BinarySearch(int[] arr, int target)
    {
        int left = 0, right = arr.Length - 1;

        while (left <= right)
        {
            int mid = left + (right - left) / 2;

            if (arr[mid] == target)
                return mid;
            else if (arr[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }

        return -1;
    }

    // Quick sort
    public void QuickSort(int[] arr, int left, int right)
    {
        if (left < right)
        {
            int pivotIndex = Partition(arr, left, right);
            QuickSort(arr, left, pivotIndex - 1);
            QuickSort(arr, pivotIndex + 1, right);
        }
    }

    private int Partition(int[] arr, int left, int right)
    {
        int pivot = arr[right];
        int i = left - 1;

        for (int j = left; j < right; j++)
        {
            if (arr[j] < pivot)
            {
                i++;
                (arr[i], arr[j]) = (arr[j], arr[i]);
            }
        }

        (arr[i + 1], arr[right]) = (arr[right], arr[i + 1]);
        return i + 1;
    }

    // Merge sort
    public void MergeSort(int[] arr, int left, int right)
    {
        if (left < right)
        {
            int mid = left + (right - left) / 2;
            MergeSort(arr, left, mid);
            MergeSort(arr, mid + 1, right);
            Merge(arr, left, mid, right);
        }
    }

    private void Merge(int[] arr, int left, int mid, int right)
    {
        int n1 = mid - left + 1;
        int n2 = right - mid;

        int[] leftArr = new int[n1];
        int[] rightArr = new int[n2];

        Array.Copy(arr, left, leftArr, 0, n1);
        Array.Copy(arr, mid + 1, rightArr, 0, n2);

        int i = 0, j = 0, k = left;

        while (i < n1 && j < n2)
        {
            if (leftArr[i] <= rightArr[j])
                arr[k++] = leftArr[i++];
            else
                arr[k++] = rightArr[j++];
        }

        while (i < n1)
            arr[k++] = leftArr[i++];

        while (j < n2)
            arr[k++] = rightArr[j++];
    }

    // DFS (Depth-First Search)
    public void DFS(Dictionary<int, List<int>> graph, int start, HashSet<int> visited)
    {
        visited.Add(start);
        Console.WriteLine(start);

        if (graph.ContainsKey(start))
        {
            foreach (var neighbor in graph[start])
            {
                if (!visited.Contains(neighbor))
                {
                    DFS(graph, neighbor, visited);
                }
            }
        }
    }

    // BFS (Breadth-First Search)
    public void BFS(Dictionary<int, List<int>> graph, int start)
    {
        var visited = new HashSet<int>();
        var queue = new Queue<int>();

        visited.Add(start);
        queue.Enqueue(start);

        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            Console.WriteLine(node);

            if (graph.ContainsKey(node))
            {
                foreach (var neighbor in graph[node])
                {
                    if (!visited.Contains(neighbor))
                    {
                        visited.Add(neighbor);
                        queue.Enqueue(neighbor);
                    }
                }
            }
        }
    }

    // LRU Cache implementation
    public class LRUCache<TKey, TValue>
    {
        private readonly int capacity;
        private readonly Dictionary<TKey, LinkedListNode<(TKey Key, TValue Value)>> cache;
        private readonly LinkedList<(TKey Key, TValue Value)> list;

        public LRUCache(int capacity)
        {
            this.capacity = capacity;
            cache = new Dictionary<TKey, LinkedListNode<(TKey, TValue)>>(capacity);
            list = new LinkedList<(TKey, TValue)>();
        }

        public TValue Get(TKey key)
        {
            if (cache.TryGetValue(key, out var node))
            {
                // Move to front (most recently used)
                list.Remove(node);
                list.AddFirst(node);
                return node.Value.Value;
            }

            return default;
        }

        public void Put(TKey key, TValue value)
        {
            if (cache.TryGetValue(key, out var existingNode))
            {
                list.Remove(existingNode);
                cache.Remove(key);
            }
            else if (cache.Count >= capacity)
            {
                // Remove least recently used
                var last = list.Last;
                list.RemoveLast();
                cache.Remove(last.Value.Key);
            }

            var newNode = list.AddFirst((key, value));
            cache[key] = newNode;
        }
    }

    // Two Sum problem
    public int[] TwoSum(int[] nums, int target)
    {
        var map = new Dictionary<int, int>();

        for (int i = 0; i < nums.Length; i++)
        {
            int complement = target - nums[i];
            if (map.ContainsKey(complement))
            {
                return new[] { map[complement], i };
            }
            map[nums[i]] = i;
        }

        return Array.Empty<int>();
    }

    // Valid parentheses
    public bool IsValidParentheses(string s)
    {
        var stack = new Stack<char>();
        var pairs = new Dictionary<char, char>
        {
            { ')', '(' },
            { '}', '{' },
            { ']', '[' }
        };

        foreach (char c in s)
        {
            if (c == '(' || c == '{' || c == '[')
            {
                stack.Push(c);
            }
            else
            {
                if (stack.Count == 0 || stack.Pop() != pairs[c])
                    return false;
            }
        }

        return stack.Count == 0;
    }

    // Longest substring without repeating characters
    public int LengthOfLongestSubstring(string s)
    {
        var charSet = new HashSet<char>();
        int left = 0, maxLength = 0;

        for (int right = 0; right < s.Length; right++)
        {
            while (charSet.Contains(s[right]))
            {
                charSet.Remove(s[left]);
                left++;
            }

            charSet.Add(s[right]);
            maxLength = Math.Max(maxLength, right - left + 1);
        }

        return maxLength;
    }
}

#endregion

#region Interview Q&A Summary

/*
 * ============================================================================
 * COMMON .NET INTERVIEW QUESTIONS - QUICK REFERENCE
 * ============================================================================
 * 
 * 1. What is the difference between value types and reference types?
 *    - Value types: stored on stack, copied by value (int, struct, enum)
 *    - Reference types: stored on heap, copied by reference (class, interface, delegate)
 * 
 * 2. Explain boxing and unboxing
 *    - Boxing: converting value type to object (heap allocation)
 *    - Unboxing: converting object back to value type
 *    - Performance cost - avoid in hot paths
 * 
 * 3. What are the generations in garbage collection?
 *    - Gen 0: short-lived objects
 *    - Gen 1: transition generation
 *    - Gen 2: long-lived objects
 * 
 * 4. IDisposable pattern and when to use it?
 *    - For unmanaged resources (file handles, database connections, etc.)
 *    - Implement Dispose() method and finalizer if needed
 *    - Use 'using' statement for automatic disposal
 * 
 * 5. Difference between abstract class and interface?
 *    - Abstract class: can have implementation, single inheritance
 *    - Interface: no implementation (C# 8.0+ allows default), multiple inheritance
 * 
 * 6. What is async/await and how does it work?
 *    - Asynchronous programming model
 *    - Doesn't block thread while waiting
 *    - Returns Task or Task<T>
 *    - Use ConfigureAwait(false) in library code
 * 
 * 7. Difference between Task and Thread?
 *    - Thread: OS-level, heavyweight, dedicated
 *    - Task: abstraction over thread, lightweight, uses thread pool
 * 
 * 8. What is dependency injection?
 *    - Design pattern for loose coupling
 *    - Lifetimes: Transient, Scoped, Singleton
 *    - Constructor injection is preferred
 * 
 * 9. LINQ deferred vs immediate execution?
 *    - Deferred: query not executed until enumerated
 *    - Immediate: ToList(), ToArray(), Count(), etc.
 * 
 * 10. What are delegates and events?
 *     - Delegate: type-safe function pointer
 *     - Event: encapsulated delegate (publisher-subscriber pattern)
 * 
 * 11. Difference between const and readonly?
 *     - const: compile-time constant, implicitly static
 *     - readonly: runtime constant, can be instance-level
 * 
 * 12. What is covariance and contravariance?
 *     - Covariance (out): can return more derived types
 *     - Contravariance (in): can accept more derived types
 * 
 * 13. How does Entity Framework Core track changes?
 *     - Change tracker monitors entity states
 *     - States: Unchanged, Added, Modified, Deleted, Detached
 *     - Use AsNoTracking() for read-only queries
 * 
 * 14. Explain middleware pipeline in ASP.NET Core
 *     - Request delegates in sequence
 *     - Each can process request and call next
 *     - Order matters!
 * 
 * 15. What is the difference between IEnumerable and IQueryable?
 *     - IEnumerable: in-memory, LINQ to Objects
 *     - IQueryable: translates to query (SQL), deferred execution
 * 
 * 16. Memory leaks in .NET - common causes?
 *     - Event handlers not unsubscribed
 *     - Static references
 *     - Timers not disposed
 *     - Captured variables in closures
 * 
 * 17. Thread safety mechanisms?
 *     - lock statement
 *     - Interlocked for atomic operations
 *     - ReaderWriterLockSlim
 *     - ConcurrentCollections
 *     - SemaphoreSlim for async
 * 
 * 18. What is reflection and when to use it?
 *     - Runtime type inspection and manipulation
 *     - Use cases: serialization, DI containers, plugins
 *     - Performance cost - cache Type and MethodInfo
 * 
 * 19. Span<T> and Memory<T> - why use them?
 *     - High-performance scenarios
 *     - Avoid heap allocations
 *     - Stack-allocated or slices of existing memory
 * 
 * 20. What are record types?
 *     - Immutable reference types with value semantics
 *     - Built-in equality comparison
 *     - With-expressions for copying
 * 
 * ============================================================================
 * KEY PERFORMANCE TIPS
 * ============================================================================
 * 
 * 1. Use StringBuilder for string concatenation in loops
 * 2. Use ValueTask when result is often synchronous
 * 3. Prefer struct for small, immutable data
 * 4. Use ArrayPool to reduce allocations
 * 5. Avoid LINQ in hot paths
 * 6. Use Span<T> for memory-intensive operations
 * 7. Disable change tracking for read-only EF queries
 * 8. Use compiled queries for repeated EF operations
 * 9. Implement proper dispose patterns
 * 10. Use concurrent collections for thread-safe scenarios
 * 
 * ============================================================================
 * BEST PRACTICES
 * ============================================================================
 * 
 * 1. Always dispose IDisposable objects (use 'using')
 * 2. Use async/await consistently (don't mix with .Result or .Wait())
 * 3. Pass CancellationToken to async methods
 * 4. Use dependency injection for loose coupling
 * 5. Prefer composition over inheritance
 * 6. Write unit tests (AAA pattern: Arrange, Act, Assert)
 * 7. Use nullable reference types (C# 8.0+)
 * 8. Never store passwords in plain text
 * 9. Parameterize SQL queries to prevent injection
 * 10. Log exceptions with full context
 * 
 * Good luck with your interview!
 */

#endregion