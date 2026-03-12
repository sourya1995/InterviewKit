// ============================================================================
// SPRING BOOT MICROSERVICES INTERVIEW PREPARATION
// For 8+ Years SDE Experience
// Spring Boot 3.x | Spring Cloud 2023.x | Java 17+
// ============================================================================
// TOPICS COVERED:
//  1.  Spring Boot Internals & Auto-Configuration
//  2.  RESTful API Design (Controllers, DTOs, Validation, HATEOAS)
//  3.  Data Layer (Spring Data JPA, Transactions, N+1, Caching)
//  4.  Service-to-Service Communication (RestTemplate, WebClient, OpenFeign)
//  5.  API Gateway & Load Balancing (Spring Cloud Gateway, Ribbon, LoadBalancer)
//  6.  Service Discovery (Eureka, Consul)
//  7.  Resilience Patterns (Circuit Breaker, Retry, Bulkhead, Rate Limiter)
//  8.  Distributed Tracing & Observability (Micrometer, Zipkin, Actuator)
//  9.  Security (JWT, OAuth2, Spring Security, mTLS)
// 10.  Messaging & Event-Driven Architecture (Kafka, RabbitMQ, Spring Events)
// 11.  Saga Pattern & Distributed Transactions
// 12.  CQRS & Event Sourcing
// 13.  Configuration Management (Config Server, Vault, Profiles)
// 14.  Containerization & Kubernetes Integration
// 15.  Testing Microservices (Unit, Integration, Contract, WireMock)
// ============================================================================
// Maven BOM (pom.xml):
//
//   <parent>
//     <groupId>org.springframework.boot</groupId>
//     <artifactId>spring-boot-starter-parent</artifactId>
//     <version>3.2.4</version>
//   </parent>
//
//   <dependencyManagement>
//     <dependencies>
//       <dependency>
//         <groupId>org.springframework.cloud</groupId>
//         <artifactId>spring-cloud-dependencies</artifactId>
//         <version>2023.0.1</version>
//         <type>pom</type>
//         <scope>import</scope>
//       </dependency>
//     </dependencies>
//   </dependencyManagement>
//
// Key starters: spring-boot-starter-web, spring-boot-starter-data-jpa,
//   spring-boot-starter-security, spring-boot-starter-actuator,
//   spring-boot-starter-validation, spring-cloud-starter-netflix-eureka-client,
//   spring-cloud-starter-gateway, spring-cloud-starter-openfeign,
//   spring-cloud-starter-circuitbreaker-resilience4j,
//   spring-kafka, spring-boot-starter-cache, micrometer-tracing-bridge-brave
// ============================================================================

package interview.springboot;

import org.springframework.boot.*;
import org.springframework.boot.autoconfigure.*;
import org.springframework.boot.autoconfigure.condition.*;
import org.springframework.boot.context.properties.*;
import org.springframework.context.annotation.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.*;
import org.springframework.validation.annotation.*;
import org.springframework.security.config.annotation.web.builders.*;
import org.springframework.security.web.*;
import org.springframework.security.oauth2.jwt.*;
import org.springframework.data.jpa.repository.*;
import org.springframework.data.repository.*;
import org.springframework.transaction.annotation.*;
import org.springframework.cache.annotation.*;
import org.springframework.cloud.openfeign.*;
import org.springframework.cloud.client.discovery.*;
import org.springframework.cloud.client.loadbalancer.*;
import org.springframework.cloud.gateway.route.*;
import org.springframework.cloud.circuitbreaker.*;
import io.github.resilience4j.circuitbreaker.annotation.*;
import io.github.resilience4j.retry.annotation.*;
import io.github.resilience4j.bulkhead.annotation.*;
import io.github.resilience4j.ratelimiter.annotation.*;
import org.springframework.kafka.annotation.*;
import org.springframework.kafka.core.*;
import org.springframework.web.reactive.function.client.*;
import reactor.core.publisher.*;
import jakarta.persistence.*;
import jakarta.validation.*;
import jakarta.validation.constraints.*;
import java.util.*;
import java.util.concurrent.*;
import java.math.*;
import java.time.*;

// ============================================================================
// SECTION 1: SPRING BOOT INTERNALS & AUTO-CONFIGURATION
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: How does Spring Boot auto-configuration work?
 * A: @SpringBootApplication = @Configuration + @ComponentScan + @EnableAutoConfiguration.
 *    Spring Boot scans META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
 *    (pre-3.x: spring.factories). Each AutoConfiguration class uses @ConditionalOn* to decide
 *    whether to apply. Only kicks in if the class is on the classpath and no user bean overrides it.
 *
 * Q: How do you exclude an auto-configuration?
 * A: @SpringBootApplication(exclude = DataSourceAutoConfiguration.class)
 *    OR spring.autoconfigure.exclude in application.properties
 *
 * Q: What is the order of property sources (highest to lowest priority)?
 * A: Command-line args > System properties > OS env vars > application-{profile}.properties
 *    > application.properties > @PropertySource annotations > Default values
 *
 * Q: @ConfigurationProperties vs @Value — when to use which?
 * A: @ConfigurationProperties for grouped/typed config beans (relaxed binding, IDE support, validation).
 *    @Value for single one-off values. @ConfigurationProperties is preferred for production code.
 *
 * Q: What is a SpringApplication and how does it bootstrap?
 * A: 1) Creates ApplicationContext (deduces type: Servlet/Reactive/None)
 *    2) Loads ApplicationContextInitializers and ApplicationListeners from spring.factories
 *    3) Runs all CommandLineRunner / ApplicationRunner beans after context refresh
 */

// --- 1a. Main Application Entry Point ---
@SpringBootApplication
@EnableFeignClients(basePackages = "interview.springboot.clients")
@EnableDiscoveryClient
public class SpringBootMicroservicesPrep {

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(SpringBootMicroservicesPrep.class);
        // Customize before run: disable banner, set profiles, add listeners
        app.setBannerMode(Banner.Mode.OFF);
        app.setAdditionalProfiles("default");
        app.run(args);
    }

    // CommandLineRunner vs ApplicationRunner:
    // CommandLineRunner: receives String[] args
    // ApplicationRunner:  receives ApplicationArguments (parsed options, non-option args)
    @Bean
    public ApplicationRunner startupRunner() {
        return args -> {
            System.out.println("Non-option args: " + args.getNonOptionArgs());
            System.out.println("Option names: " + args.getOptionNames());
        };
    }
}

// --- 1b. Custom Auto-Configuration ---
// INTERVIEWER TRAP: How do you write your own auto-configuration (e.g., for a shared library)?
// Step 1: Create the configuration class with @ConditionalOn* guards
// Step 2: Register in META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports
@AutoConfiguration
@ConditionalOnClass(name = "com.example.SomeLibrary")           // only if class on classpath
@ConditionalOnMissingBean(type = "com.example.SomeLibraryBean") // only if user hasn't defined one
@ConditionalOnProperty(prefix = "mylib", name = "enabled", havingValue = "true", matchIfMissing = true)
class MyLibraryAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public Object someLibraryBean() {
        return new Object(); // replaced with real library bean
    }
}

// --- 1c. @ConfigurationProperties (Relaxed Binding + Validation) ---
/**
 * INTERVIEW TRAP: What is relaxed binding?
 * A: Spring Boot maps these equivalent forms to the same property:
 *    my.service.max-connections, my.service.maxConnections, MY_SERVICE_MAX_CONNECTIONS
 *    This is ONLY available with @ConfigurationProperties, NOT with @Value.
 */
@Configuration
@ConfigurationProperties(prefix = "app.service")
@Validated
class AppServiceProperties {

    @NotBlank
    private String name;

    @Min(1) @Max(100)
    private int maxConnections = 10;

    @NotNull
    private Duration timeout = Duration.ofSeconds(5);

    private Map<String, String> headers = new HashMap<>();

    // Getters and setters required (or use record / @ConstructorBinding in Boot 3.x)
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getMaxConnections() { return maxConnections; }
    public void setMaxConnections(int maxConnections) { this.maxConnections = maxConnections; }
    public Duration getTimeout() { return timeout; }
    public void setTimeout(Duration timeout) { this.timeout = timeout; }
    public Map<String, String> getHeaders() { return headers; }
    public void setHeaders(Map<String, String> headers) { this.headers = headers; }
}

// ============================================================================
// SECTION 2: RESTFUL API DESIGN — Controllers, DTOs, Validation, Exception Handling
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: @RestController vs @Controller?
 * A: @RestController = @Controller + @ResponseBody on every method. No need for @ResponseBody per method.
 *
 * Q: What HTTP status code should POST return on success?
 * A: 201 Created with Location header pointing to the new resource. Use ResponseEntity.created(uri).build()
 *
 * Q: How does @Valid vs @Validated differ?
 * A: @Valid is standard JSR-380. @Validated is Spring's extension supporting group validation and
 *    method-level validation on non-controller beans.
 *
 * Q: How do you handle exceptions globally?
 * A: @RestControllerAdvice + @ExceptionHandler methods. Returns structured error response.
 *    Avoids duplicating try-catch in every controller.
 *
 * Q: How do you version a REST API?
 * A: URI versioning (/v1/users), Header versioning (Accept: application/vnd.company.v1+json),
 *    Query param versioning (?version=1). URI is most common and discoverable.
 *
 * Q: What is HATEOAS and when should you use it?
 * A: Hypermedia As The Engine Of Application State — responses include links to related actions.
 *    Use Spring HATEOAS. Good for mature REST APIs; often skipped for internal microservices.
 */

// --- 2a. Request/Response DTOs ---
record CreateOrderRequest(
    @NotBlank(message = "Customer ID is required")
    String customerId,

    @NotEmpty(message = "Items cannot be empty")
    @Valid
    List<OrderItemRequest> items,

    @NotNull
    @Future(message = "Delivery date must be in the future")
    LocalDate deliveryDate
) {}

record OrderItemRequest(
    @NotBlank String productId,
    @Min(1) int quantity,
    @DecimalMin("0.01") BigDecimal unitPrice
) {}

record OrderResponse(
    String orderId,
    String customerId,
    String status,
    BigDecimal totalAmount,
    LocalDateTime createdAt
) {}

// Standardized error response — ALWAYS define this in interviews
record ApiError(
    int status,
    String error,
    String message,
    String path,
    LocalDateTime timestamp,
    Map<String, String> fieldErrors
) {}

// --- 2b. REST Controller (best practices) ---
@RestController
@RequestMapping("/api/v1/orders")
@Validated
class OrderController {

    private final OrderService orderService;

    public OrderController(OrderService orderService) { // Constructor injection — preferred
        this.orderService = orderService;
    }

    @PostMapping
    public ResponseEntity<OrderResponse> createOrder(
            @Valid @RequestBody CreateOrderRequest request,
            UriComponentsBuilder uriBuilder) {

        OrderResponse response = orderService.createOrder(request);
        URI location = uriBuilder.path("/api/v1/orders/{id}")
                .buildAndExpand(response.orderId())
                .toUri();
        return ResponseEntity.created(location).body(response);  // 201 + Location header
    }

    @GetMapping("/{orderId}")
    public ResponseEntity<OrderResponse> getOrder(@PathVariable String orderId) {
        return ResponseEntity.ok(orderService.getOrder(orderId));
    }

    @GetMapping
    public ResponseEntity<Page<OrderResponse>> listOrders(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(required = false) String status) {
        return ResponseEntity.ok(orderService.listOrders(page, size, status));
    }

    @PatchMapping("/{orderId}/status")
    public ResponseEntity<OrderResponse> updateStatus(
            @PathVariable String orderId,
            @RequestBody @Valid UpdateStatusRequest request) {
        return ResponseEntity.ok(orderService.updateStatus(orderId, request));
    }

    @DeleteMapping("/{orderId}")
    public ResponseEntity<Void> cancelOrder(@PathVariable String orderId) {
        orderService.cancelOrder(orderId);
        return ResponseEntity.noContent().build(); // 204
    }
}

// Placeholder records for compilation
record UpdateStatusRequest(@NotBlank String status) {}
record Page<T>(List<T> content, int totalPages, long totalElements) {}

// --- 2c. Global Exception Handler ---
@RestControllerAdvice
class GlobalExceptionHandler {

    // Validation errors (@Valid failures)
    @ExceptionHandler(org.springframework.web.bind.MethodArgumentNotValidException.class)
    public ResponseEntity<ApiError> handleValidation(
            org.springframework.web.bind.MethodArgumentNotValidException ex,
            jakarta.servlet.http.HttpServletRequest request) {

        Map<String, String> fieldErrors = new LinkedHashMap<>();
        ex.getBindingResult().getFieldErrors()
                .forEach(fe -> fieldErrors.put(fe.getField(), fe.getDefaultMessage()));

        ApiError error = new ApiError(
                HttpStatus.BAD_REQUEST.value(), "Validation Failed",
                "One or more fields have invalid values",
                request.getRequestURI(), LocalDateTime.now(), fieldErrors);
        return ResponseEntity.badRequest().body(error);
    }

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ApiError> handleNotFound(
            ResourceNotFoundException ex,
            jakarta.servlet.http.HttpServletRequest request) {
        ApiError error = new ApiError(
                HttpStatus.NOT_FOUND.value(), "Not Found",
                ex.getMessage(), request.getRequestURI(),
                LocalDateTime.now(), Collections.emptyMap());
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(error);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleGeneral(
            Exception ex, jakarta.servlet.http.HttpServletRequest request) {
        ApiError error = new ApiError(
                HttpStatus.INTERNAL_SERVER_ERROR.value(), "Internal Server Error",
                "An unexpected error occurred", request.getRequestURI(),
                LocalDateTime.now(), Collections.emptyMap());
        return ResponseEntity.internalServerError().body(error);
    }
}

class ResourceNotFoundException extends RuntimeException {
    public ResourceNotFoundException(String message) { super(message); }
}

// ============================================================================
// SECTION 3: DATA LAYER — JPA, Transactions, N+1 Problem, Caching
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: What is the N+1 problem in JPA and how do you fix it?
 * A: When you fetch a list of N entities each with a lazy-loaded association,
 *    JPA fires 1 query to get the list + N queries for each association = N+1 queries.
 *    Fix: JOIN FETCH in JPQL, @EntityGraph, or @BatchSize(size=50).
 *
 * Q: What is the difference between EAGER and LAZY loading?
 * A: EAGER: loads association immediately with parent (JOIN or extra SELECT).
 *    LAZY: loads only when accessed. OneToMany/ManyToMany are LAZY by default.
 *    TRAP: EAGER can cause massive data loads; LAZY can cause LazyInitializationException
 *    if accessed outside a transaction.
 *
 * Q: @Transactional propagation types?
 * A: REQUIRED (default): join existing or create new.
 *    REQUIRES_NEW: always suspend existing, create new (separate tx, own commit/rollback).
 *    NESTED: nested savepoint within outer tx.
 *    SUPPORTS: use existing if present, else non-transactional.
 *    NOT_SUPPORTED: suspend existing, run non-transactionally.
 *    NEVER: throw if transaction exists.
 *    MANDATORY: throw if NO transaction exists.
 *
 * Q: @Transactional on private methods — does it work?
 * A: NO. Spring AOP proxies only intercept public method calls from OUTSIDE the bean.
 *    Self-invocation (this.method()) bypasses the proxy entirely.
 *
 * Q: What is Optimistic vs Pessimistic Locking?
 * A: Optimistic: @Version field; throws OptimisticLockException on conflict (no DB lock held).
 *    Pessimistic: SELECT ... FOR UPDATE; holds DB lock (PESSIMISTIC_WRITE / PESSIMISTIC_READ).
 *    Use Optimistic for low-contention; Pessimistic for high-contention or financial data.
 */

// --- 3a. Entity Design ---
@Entity
@Table(name = "orders", indexes = {
    @Index(name = "idx_order_customer", columnList = "customer_id"),
    @Index(name = "idx_order_status", columnList = "status")
})
class Order {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID) // UUID strategy in JPA 3.1+
    private String id;

    @Column(name = "customer_id", nullable = false)
    private String customerId;

    @Enumerated(EnumType.STRING) // STRING not ORDINAL — adding enum values won't break existing data
    @Column(nullable = false)
    private OrderStatus status;

    @Column(precision = 19, scale = 4)
    private BigDecimal totalAmount;

    // FetchType.LAZY prevents N+1 — use JOIN FETCH when needed
    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    private List<OrderItem> items = new ArrayList<>();

    @Version // Optimistic locking — auto-incremented on each update
    private Long version;

    @CreationTimestamp
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    // Domain logic inside entity (DDD-style)
    public void addItem(OrderItem item) {
        items.add(item);
        item.setOrder(this);
        recalculateTotal();
    }

    public void cancel() {
        if (status != OrderStatus.PENDING) {
            throw new IllegalStateException("Only PENDING orders can be cancelled");
        }
        this.status = OrderStatus.CANCELLED;
    }

    private void recalculateTotal() {
        this.totalAmount = items.stream()
                .map(i -> i.getUnitPrice().multiply(BigDecimal.valueOf(i.getQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    // Getters/setters omitted for brevity
    public String getId() { return id; }
    public String getCustomerId() { return customerId; }
    public void setCustomerId(String customerId) { this.customerId = customerId; }
    public OrderStatus getStatus() { return status; }
    public void setStatus(OrderStatus status) { this.status = status; }
    public List<OrderItem> getItems() { return items; }
    public BigDecimal getTotalAmount() { return totalAmount; }
}

@Entity
@Table(name = "order_items")
class OrderItem {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "order_id", nullable = false)
    private Order order;

    private String productId;
    private int quantity;

    @Column(precision = 19, scale = 4)
    private BigDecimal unitPrice;

    public void setOrder(Order order) { this.order = order; }
    public Order getOrder() { return order; }
    public String getProductId() { return productId; }
    public int getQuantity() { return quantity; }
    public BigDecimal getUnitPrice() { return unitPrice; }
}

enum OrderStatus { PENDING, CONFIRMED, SHIPPED, DELIVERED, CANCELLED }

// --- 3b. Repository — @EntityGraph to fix N+1 ---
@org.springframework.stereotype.Repository
interface OrderRepository extends JpaRepository<Order, String> {

    // N+1 FIX: JOIN FETCH loads items in the same query
    @Query("SELECT o FROM Order o JOIN FETCH o.items WHERE o.customerId = :customerId")
    List<Order> findByCustomerIdWithItems(@Param("customerId") String customerId);

    // @EntityGraph alternative — no JPQL needed, declarative
    @EntityGraph(attributePaths = {"items"})
    List<Order> findByStatus(OrderStatus status);

    // Projection — only fetches required columns (avoids loading the full entity)
    @Query("SELECT o.id as orderId, o.status as status, o.totalAmount as totalAmount FROM Order o WHERE o.customerId = :customerId")
    List<OrderSummary> findSummariesByCustomerId(@Param("customerId") String customerId);

    // Pessimistic lock — prevents concurrent modification
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT o FROM Order o WHERE o.id = :id")
    Optional<Order> findByIdWithLock(@Param("id") String id);

    // Pageable support
    org.springframework.data.domain.Page<Order> findByCustomerId(
            String customerId, org.springframework.data.domain.Pageable pageable);
}

// Projection interface — Spring Data generates a proxy; only mapped columns are SELECTed
interface OrderSummary {
    String getOrderId();
    OrderStatus getStatus();
    BigDecimal getTotalAmount();
}

// --- 3c. Service with Transactional Patterns ---
@org.springframework.stereotype.Service
@Transactional(readOnly = true) // Default all methods read-only — better performance (no flush)
class OrderService {

    private final OrderRepository orderRepository;

    public OrderService(OrderRepository orderRepository) {
        this.orderRepository = orderRepository;
    }

    @Transactional // Overrides readOnly=true for write operations
    public OrderResponse createOrder(CreateOrderRequest request) {
        Order order = new Order();
        order.setCustomerId(request.customerId());
        order.setStatus(OrderStatus.PENDING);
        // Map items...
        Order saved = orderRepository.save(order);
        return mapToResponse(saved);
    }

    public OrderResponse getOrder(String orderId) {
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new ResourceNotFoundException("Order not found: " + orderId));
        return mapToResponse(order);
    }

    public Page<OrderResponse> listOrders(int page, int size, String status) {
        // implementation
        return new Page<>(Collections.emptyList(), 0, 0);
    }

    @Transactional
    public OrderResponse updateStatus(String orderId, UpdateStatusRequest request) {
        // Uses optimistic locking via @Version — will throw if concurrent modification
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new ResourceNotFoundException("Order not found: " + orderId));
        order.setStatus(OrderStatus.valueOf(request.status()));
        return mapToResponse(order); // No explicit save — dirty checking handles it
    }

    @Transactional
    public void cancelOrder(String orderId) {
        Order order = orderRepository.findByIdWithLock(orderId) // Pessimistic lock for cancel
                .orElseThrow(() -> new ResourceNotFoundException("Order not found: " + orderId));
        order.cancel();
    }

    // TRAP: @Transactional(propagation = REQUIRES_NEW) use case
    // Audit logging must persist even if the outer transaction rolls back
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void saveAuditLog(String action, String entityId) {
        // Saved in its own transaction — independent of caller's tx
    }

    private OrderResponse mapToResponse(Order order) {
        return new OrderResponse(
                order.getId(), order.getCustomerId(),
                order.getStatus().name(), order.getTotalAmount(),
                LocalDateTime.now());
    }
}

// --- 3d. Caching with Spring Cache ---
@org.springframework.stereotype.Service
@CacheConfig(cacheNames = "products") // Default cache name for all methods
class ProductCacheService {

    // @Cacheable: cache result; skip method if cache hit
    @Cacheable(key = "#productId", unless = "#result == null")
    public Object getProduct(String productId) {
        // DB call — only executed on cache miss
        return null;
    }

    // @CachePut: ALWAYS executes method and updates cache
    @CachePut(key = "#result.id")
    public Object updateProduct(Object product) {
        return product; // result is cached
    }

    // @CacheEvict: removes from cache
    @CacheEvict(key = "#productId")
    public void deleteProduct(String productId) { /* delete from DB */ }

    // @Caching: combine multiple cache operations on one method
    @Caching(
        evict = { @CacheEvict(cacheNames = "products", key = "#productId") },
        put   = { @CachePut(cacheNames = "product-list", key = "'all'") }
    )
    public Object refreshProduct(String productId) { return null; }
}

// ============================================================================
// SECTION 4: SERVICE-TO-SERVICE COMMUNICATION
// RestTemplate (legacy) | WebClient (reactive) | OpenFeign (declarative)
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: RestTemplate vs WebClient vs OpenFeign — when to use each?
 * A: RestTemplate: Legacy blocking HTTP client. Deprecated in Boot 3.x. Avoid for new code.
 *    WebClient: Non-blocking reactive client. Best for high-throughput or reactive apps.
 *    OpenFeign: Declarative HTTP client. Best for microservice-to-microservice calls —
 *               interface-based, integrates with Eureka (service discovery) and Resilience4j.
 *
 * Q: How does OpenFeign integrate with load balancing?
 * A: When combined with @EnableDiscoveryClient, Feign resolves service names (e.g., "inventory-service")
 *    via Eureka/Consul and uses Spring Cloud LoadBalancer (round-robin by default) to pick an instance.
 *
 * Q: How do you propagate headers (e.g., auth token, trace ID) in Feign calls?
 * A: Implement RequestInterceptor — called for every Feign request.
 *
 * Q: What is the difference between synchronous and asynchronous inter-service communication?
 * A: Synchronous (REST/Feign/gRPC): simple, immediate response, but creates temporal coupling —
 *    if inventory-service is down, order-service fails too.
 *    Asynchronous (Kafka/RabbitMQ): decoupled, resilient, but eventual consistency only.
 */

// --- 4a. OpenFeign Client ---
@FeignClient(
    name = "inventory-service",           // Resolves via service discovery
    fallback = InventoryClientFallback.class  // Resilience4j fallback
)
interface InventoryClient {

    @GetMapping("/api/v1/inventory/{productId}")
    InventoryResponse checkInventory(@PathVariable("productId") String productId);

    @PostMapping("/api/v1/inventory/reserve")
    ReservationResponse reserveStock(@RequestBody ReserveStockRequest request);

    @PutMapping("/api/v1/inventory/{productId}/release")
    void releaseReservation(
            @PathVariable("productId") String productId,
            @RequestParam("quantity") int quantity);
}

// Placeholder types
record InventoryResponse(String productId, int availableQuantity, boolean inStock) {}
record ReservationResponse(String reservationId, boolean success, String reason) {}
record ReserveStockRequest(String productId, int quantity, String orderId) {}

// Feign Fallback — called when circuit is open or target returns error
@org.springframework.stereotype.Component
class InventoryClientFallback implements InventoryClient {

    @Override
    public InventoryResponse checkInventory(String productId) {
        // Return a safe default — indicate stock unknown
        return new InventoryResponse(productId, 0, false);
    }

    @Override
    public ReservationResponse reserveStock(ReserveStockRequest request) {
        return new ReservationResponse(null, false, "Inventory service unavailable");
    }

    @Override
    public void releaseReservation(String productId, int quantity) {
        // Log for reconciliation — can't do anything here
    }
}

// --- 4b. Feign RequestInterceptor — propagate auth headers ---
@Configuration
class FeignAuthInterceptor implements feign.RequestInterceptor {

    @Override
    public void apply(feign.RequestTemplate template) {
        // Extract JWT from SecurityContext and forward it
        var auth = org.springframework.security.core.context.SecurityContextHolder
                .getContext().getAuthentication();
        if (auth != null && auth.getCredentials() instanceof String token) {
            template.header("Authorization", "Bearer " + token);
        }
        // Forward trace headers for distributed tracing
        template.header("X-Request-Id", UUID.randomUUID().toString());
    }
}

// --- 4c. WebClient (Reactive / Non-blocking) ---
@Configuration
class WebClientConfig {

    @Bean
    @LoadBalanced // Makes WebClient use service discovery + load balancing
    public WebClient.Builder loadBalancedWebClientBuilder() {
        return WebClient.builder()
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .codecs(c -> c.defaultCodecs().maxInMemorySize(1024 * 1024)); // 1MB limit
    }
}

@org.springframework.stereotype.Service
class PaymentServiceClient {

    private final WebClient webClient;

    public PaymentServiceClient(WebClient.Builder builder) {
        this.webClient = builder.baseUrl("http://payment-service").build();
    }

    // Reactive call — returns Mono (0 or 1 result)
    public Mono<PaymentResponse> processPayment(PaymentRequest request) {
        return webClient.post()
                .uri("/api/v1/payments")
                .bodyValue(request)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError,
                        resp -> resp.bodyToMono(String.class)
                                .flatMap(body -> Mono.error(new IllegalArgumentException("Payment rejected: " + body))))
                .onStatus(HttpStatusCode::is5xxServerError,
                        resp -> Mono.error(new RuntimeException("Payment service error")))
                .bodyToMono(PaymentResponse.class)
                .timeout(Duration.ofSeconds(10))
                .retry(2);
    }

    // Flux — for streaming multiple results
    public Flux<PaymentResponse> getPaymentHistory(String customerId) {
        return webClient.get()
                .uri("/api/v1/payments?customerId={id}", customerId)
                .retrieve()
                .bodyToFlux(PaymentResponse.class);
    }
}

// Placeholder types
record PaymentRequest(String orderId, BigDecimal amount, String currency) {}
record PaymentResponse(String paymentId, String status, BigDecimal amount) {}

// ============================================================================
// SECTION 5: API GATEWAY & LOAD BALANCING
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: What is an API Gateway and why do microservices need one?
 * A: Single entry point for all client requests. Responsibilities:
 *    - Request routing (to the right microservice)
 *    - Authentication/Authorization (JWT validation before reaching services)
 *    - Rate limiting (prevent abuse)
 *    - Request/Response transformation
 *    - Aggregation (combine responses from multiple services)
 *    - SSL termination
 *    - Load balancing
 *    Without it, clients need to know all service addresses and handle auth themselves.
 *
 * Q: Spring Cloud Gateway vs Zuul?
 * A: Zuul 1.x is blocking (Servlet-based). Spring Cloud Gateway is non-blocking (WebFlux/Netty).
 *    Gateway has better performance, first-class support in Spring Cloud 2023.x.
 *    Zuul 2 is also non-blocking but not in Spring Cloud ecosystem.
 *
 * Q: How do you implement rate limiting in Spring Cloud Gateway?
 * A: RequestRateLimiterGatewayFilterFactory with Redis-backed token bucket algorithm.
 *    Configure via RedisRateLimiter bean or YAML.
 *
 * Q: What is the difference between Gateway Predicates and Filters?
 * A: Predicates: conditions to match a route (path, method, header, query param, etc.)
 *    Filters: modify request/response (add headers, rewrite path, retry, rate limit, etc.)
 */

// --- 5a. Programmatic Route Configuration ---
@Configuration
class GatewayRoutesConfig {

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()

            // Order Service route with path rewrite and JWT auth filter
            .route("order-service", r -> r
                .path("/api/v1/orders/**")
                .filters(f -> f
                    .rewritePath("/api/v1/orders/(?<segment>.*)", "/orders/${segment}")
                    .addRequestHeader("X-Gateway-Hop", "true")
                    .retry(config -> config
                            .setRetries(3)
                            .setStatuses(HttpStatus.SERVICE_UNAVAILABLE)
                            .setMethods(HttpMethod.GET))
                    .circuitBreaker(config -> config
                            .setName("order-service-cb")
                            .setFallbackUri("forward:/fallback/orders"))
                )
                .uri("lb://order-service")) // "lb://" triggers load balancing via discovery

            // Inventory service — only GET methods (read-only gateway rule)
            .route("inventory-read", r -> r
                .path("/api/v1/inventory/**")
                .and().method(HttpMethod.GET)
                .filters(f -> f
                    .requestRateLimiter(config -> config
                            .setRateLimiter(redisRateLimiter())
                            .setKeyResolver(ipKeyResolver())))
                .uri("lb://inventory-service"))

            .build();
    }

    // Redis-backed rate limiter: 10 requests/second, burst up to 20
    @Bean
    public org.springframework.cloud.gateway.filter.ratelimit.RedisRateLimiter redisRateLimiter() {
        return new org.springframework.cloud.gateway.filter.ratelimit.RedisRateLimiter(10, 20);
    }

    // Rate limit by client IP
    @Bean
    public org.springframework.cloud.gateway.filter.ratelimit.KeyResolver ipKeyResolver() {
        return exchange -> Mono.just(
                Objects.requireNonNull(exchange.getRequest().getRemoteAddress())
                        .getAddress().getHostAddress()
        );
    }
}

// --- 5b. Global Gateway Filter (runs for ALL routes) ---
@Component
class AuthenticationGatewayFilter implements
        org.springframework.cloud.gateway.filter.GlobalFilter,
        org.springframework.core.Ordered {

    private final JwtDecoder jwtDecoder;
    private static final List<String> PUBLIC_PATHS = List.of("/auth/", "/actuator/health");

    public AuthenticationGatewayFilter(JwtDecoder jwtDecoder) {
        this.jwtDecoder = jwtDecoder;
    }

    @Override
    public Mono<Void> filter(org.springframework.web.server.ServerWebExchange exchange,
                             org.springframework.cloud.gateway.filter.GatewayFilterChain chain) {

        String path = exchange.getRequest().getPath().value();

        // Skip auth for public endpoints
        if (PUBLIC_PATHS.stream().anyMatch(path::startsWith)) {
            return chain.filter(exchange);
        }

        String authHeader = exchange.getRequest().getHeaders().getFirst("Authorization");
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }

        try {
            Jwt jwt = jwtDecoder.decode(authHeader.substring(7));
            // Forward user info to downstream services as headers
            var mutatedRequest = exchange.getRequest().mutate()
                    .header("X-User-Id", jwt.getSubject())
                    .header("X-User-Roles", String.join(",",
                            jwt.getClaimAsStringList("roles")))
                    .build();
            return chain.filter(exchange.mutate().request(mutatedRequest).build());
        } catch (Exception e) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }
    }

    @Override
    public int getOrder() { return -100; } // Run before other filters
}

// ============================================================================
// SECTION 6: SERVICE DISCOVERY (Eureka / Consul)
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: How does Eureka service discovery work?
 * A: Eureka Server: registry of all service instances (name -> IP:port).
 *    Eureka Client: on startup, registers itself. Sends heartbeats every 30s.
 *    If no heartbeat for 90s (3 missed), Eureka removes the instance.
 *    Clients cache the registry locally — still work if Eureka is temporarily down (self-preservation).
 *
 * Q: What is Eureka's self-preservation mode?
 * A: If Eureka sees > 15% of clients stopped sending heartbeats in 1 minute, it assumes a
 *    network partition (not service failure) and STOPS evicting instances. Prevents mass
 *    deregistration during network blips. Disable in dev: eureka.server.enable-self-preservation=false
 *
 * Q: Client-side vs Server-side load balancing?
 * A: Server-side: load balancer (nginx, AWS ALB) sits in front, client doesn't know about instances.
 *    Client-side: client fetches all instances from registry, picks one itself (Spring Cloud LoadBalancer).
 *    Client-side: lower latency (no extra hop), more complex client, but better for microservices.
 *
 * Q: How does @LoadBalanced work?
 * A: It's a qualifier annotation. Spring creates a special interceptor/filter that intercepts
 *    HTTP calls with service names (http://service-name/...) and replaces them with actual
 *    IP:port using the registry.
 */

// --- 6a. Eureka Server Setup ---
// @SpringBootApplication
// @EnableEurekaServer  ← add this to a dedicated server application
// class EurekaServerApplication { public static void main(String[] a) { SpringApplication.run(...); } }

// application.yml for Eureka Server:
// eureka:
//   client:
//     register-with-eureka: false  ← server doesn't register itself
//     fetch-registry: false
//   server:
//     enable-self-preservation: false  ← disable in dev/test

// --- 6b. Eureka Client Configuration ---
// application.yml for each microservice:
// spring:
//   application:
//     name: order-service          ← this is the service name other services use
// eureka:
//   client:
//     service-url:
//       defaultZone: http://localhost:8761/eureka/
//   instance:
//     prefer-ip-address: true      ← register with IP, not hostname
//     lease-renewal-interval-in-seconds: 10
//     lease-expiration-duration-in-seconds: 30

// --- 6c. Programmatic Service Discovery ---
@org.springframework.stereotype.Service
class ServiceDiscoveryHelper {

    private final org.springframework.cloud.client.discovery.DiscoveryClient discoveryClient;

    public ServiceDiscoveryHelper(org.springframework.cloud.client.discovery.DiscoveryClient discoveryClient) {
        this.discoveryClient = discoveryClient;
    }

    public List<String> getAllServiceUrls(String serviceName) {
        return discoveryClient.getInstances(serviceName)
                .stream()
                .map(instance -> instance.getUri().toString())
                .toList();
    }

    public void printAllServices() {
        System.out.println("Registered services: " + discoveryClient.getServices());
        discoveryClient.getInstances("inventory-service")
                .forEach(inst -> System.out.printf("  %s:%d (metadata: %s)%n",
                        inst.getHost(), inst.getPort(), inst.getMetadata()));
    }
}

// ============================================================================
// SECTION 7: RESILIENCE PATTERNS — Circuit Breaker, Retry, Bulkhead, Rate Limiter
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: What is a Circuit Breaker and what problem does it solve?
 * A: Prevents cascading failures. Like an electrical circuit breaker:
 *    CLOSED (normal): requests pass through; failures counted.
 *    OPEN (tripped): all requests fail immediately (no waiting) — fast fail.
 *    HALF_OPEN (recovery): limited requests let through to test if service recovered.
 *    Without it: slow service causes thread pool exhaustion in calling service → cascading failure.
 *
 * Q: Resilience4j CircuitBreaker configuration parameters?
 * A: slidingWindowSize: number of calls or seconds to evaluate (COUNT_BASED or TIME_BASED)
 *    failureRateThreshold: % of failures to trip the breaker (default 50%)
 *    waitDurationInOpenState: how long to stay OPEN before trying HALF_OPEN (default 60s)
 *    permittedNumberOfCallsInHalfOpenState: test calls in HALF_OPEN (default 10)
 *    minimumNumberOfCalls: minimum calls before calculating failure rate (default 100)
 *
 * Q: Retry vs Circuit Breaker — how do they interact?
 * A: Retry should be INSIDE Circuit Breaker (inner decorator). Retry first; if still failing,
 *    CircuitBreaker trips. If CB is OPEN, Retry should NOT retry (waste of time).
 *    Order: CircuitBreaker > Retry > Bulkhead > RateLimiter (outer to inner)
 *
 * Q: What is Bulkhead pattern?
 * A: Isolates failures. Two types:
 *    SemaphoreBulkhead: limits concurrent calls (semaphore permits).
 *    ThreadPoolBulkhead: each integration has its own thread pool — isolation from main pool.
 *    If inventory-service is slow, it doesn't exhaust threads for payment-service.
 *
 * Q: What is a fallback?
 * A: Alternative behavior when primary fails. Should return cached data, default value,
 *    or a graceful degraded response — NOT throw exceptions to the caller.
 */

// --- 7a. Circuit Breaker with Resilience4j ---
@org.springframework.stereotype.Service
class ResilientInventoryService {

    private final InventoryClient inventoryClient;

    public ResilientInventoryService(InventoryClient inventoryClient) {
        this.inventoryClient = inventoryClient;
    }

    // CircuitBreaker + Retry + Bulkhead stacked
    // fallbackMethod MUST have same signature + Throwable parameter
    @CircuitBreaker(name = "inventoryCB", fallbackMethod = "inventoryFallback")
    @Retry(name = "inventoryRetry")
    @Bulkhead(name = "inventoryBulkhead")
    @RateLimiter(name = "inventoryRL")
    public InventoryResponse checkInventorySafely(String productId) {
        return inventoryClient.checkInventory(productId);
    }

    // Fallback — same return type, extra Throwable param
    public InventoryResponse inventoryFallback(String productId, Throwable ex) {
        System.err.println("Circuit breaker fallback for product: " + productId + ", reason: " + ex.getMessage());
        return new InventoryResponse(productId, 0, false); // Safe default
    }

    // Fallback for specific exception types
    public InventoryResponse inventoryFallback(String productId,
            io.github.resilience4j.circuitbreaker.CallNotPermittedException ex) {
        System.err.println("Circuit is OPEN for inventory service — fast failing");
        return new InventoryResponse(productId, -1, false);
    }
}

// --- 7b. Resilience4j Configuration (application.yml equivalent in code) ---
// application.yml:
// resilience4j:
//   circuitbreaker:
//     instances:
//       inventoryCB:
//         sliding-window-type: COUNT_BASED
//         sliding-window-size: 10
//         failure-rate-threshold: 50
//         wait-duration-in-open-state: 30s
//         permitted-number-of-calls-in-half-open-state: 3
//         minimum-number-of-calls: 5
//         record-exceptions:
//           - java.io.IOException
//           - feign.FeignException.ServiceUnavailable
//   retry:
//     instances:
//       inventoryRetry:
//         max-attempts: 3
//         wait-duration: 500ms
//         retry-exceptions:
//           - java.io.IOException
//         ignore-exceptions:
//           - interview.springboot.ResourceNotFoundException
//   bulkhead:
//     instances:
//       inventoryBulkhead:
//         max-concurrent-calls: 10
//         max-wait-duration: 100ms
//   rate-limiter:
//     instances:
//       inventoryRL:
//         limit-for-period: 50
//         limit-refresh-period: 1s
//         timeout-duration: 100ms

// --- 7c. Programmatic Resilience4j (when annotations aren't enough) ---
@org.springframework.stereotype.Component
class ProgrammaticCircuitBreaker {

    private final io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry registry;

    public ProgrammaticCircuitBreaker(io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry registry) {
        this.registry = registry;
    }

    public <T> T executeWithCB(String cbName, java.util.function.Supplier<T> supplier, java.util.function.Supplier<T> fallback) {
        var cb = registry.circuitBreaker(cbName);
        return io.github.resilience4j.circuitbreaker.CircuitBreaker
                .decorateSupplier(cb, supplier)
                .get();
    }

    public void monitorCircuitBreaker(String cbName) {
        var cb = registry.circuitBreaker(cbName);
        cb.getEventPublisher()
                .onStateTransition(e -> System.out.printf(
                        "CB [%s] state: %s -> %s%n",
                        cbName, e.getStateTransition().getFromState(),
                        e.getStateTransition().getToState()));
        cb.getEventPublisher()
                .onFailureRateExceeded(e -> System.out.printf(
                        "CB [%s] failure rate: %.2f%n", cbName, e.getFailureRate()));
    }
}

// ============================================================================
// SECTION 8: OBSERVABILITY — Distributed Tracing, Metrics, Actuator
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: What is distributed tracing and why is it essential in microservices?
 * A: A single user request spans multiple microservices. Without tracing, you can't correlate
 *    logs across services. Distributed tracing assigns a TraceId to each request (shared across
 *    all hops) and a SpanId to each individual service call.
 *    Stack: Micrometer Tracing (instrumentation) + Brave/OpenTelemetry (implementation) + Zipkin/Jaeger (UI).
 *
 * Q: What headers does B3 propagation use?
 * A: X-B3-TraceId, X-B3-SpanId, X-B3-ParentSpanId, X-B3-Sampled
 *    These are automatically propagated by Spring Cloud Sleuth / Micrometer Tracing.
 *
 * Q: What is Spring Boot Actuator and which endpoints are important?
 * A: Exposes operational info: /actuator/health (liveness/readiness), /actuator/metrics,
 *    /actuator/info, /actuator/env, /actuator/loggers, /actuator/circuitbreakers,
 *    /actuator/prometheus (for Prometheus scraping).
 *
 * Q: Liveness vs Readiness probes (Kubernetes)?
 * A: Liveness: Is the app alive? If fails, Kubernetes RESTARTS the pod.
 *    Readiness: Is the app ready to receive traffic? If fails, Kubernetes REMOVES from load balancer.
 *    In Spring Boot Actuator:
 *    /actuator/health/liveness  → LivenessStateHealthIndicator
 *    /actuator/health/readiness → ReadinessStateHealthIndicator
 *
 * Q: What is the difference between metrics, logs, and traces?
 * A: Metrics: numeric aggregates over time (CPU, request rate, error rate) — for alerting.
 *    Logs: discrete events with context — for debugging.
 *    Traces: request journey across services — for performance analysis.
 *    Together they form the "three pillars of observability".
 */

// --- 8a. Custom Health Indicator ---
@org.springframework.stereotype.Component
class KafkaHealthIndicator implements org.springframework.boot.actuate.health.HealthIndicator {

    private final KafkaTemplate<String, String> kafkaTemplate;

    public KafkaHealthIndicator(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Override
    public org.springframework.boot.actuate.health.Health health() {
        try {
            // Check if we can access Kafka metadata
            kafkaTemplate.getProducerFactory().createProducer().partitionsFor("health-check");
            return org.springframework.boot.actuate.health.Health.up()
                    .withDetail("kafka", "connected")
                    .build();
        } catch (Exception e) {
            return org.springframework.boot.actuate.health.Health.down()
                    .withDetail("kafka", "disconnected")
                    .withException(e)
                    .build();
        }
    }
}

// --- 8b. Custom Metrics with Micrometer ---
@org.springframework.stereotype.Service
class OrderMetricsService {

    private final io.micrometer.core.instrument.MeterRegistry meterRegistry;
    private final io.micrometer.core.instrument.Counter orderCreatedCounter;
    private final io.micrometer.core.instrument.Timer orderProcessingTimer;
    private final io.micrometer.core.instrument.Gauge pendingOrdersGauge;
    private final java.util.concurrent.atomic.AtomicInteger pendingOrderCount;

    public OrderMetricsService(io.micrometer.core.instrument.MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.pendingOrderCount = new java.util.concurrent.atomic.AtomicInteger(0);

        // Counter: monotonically increasing
        this.orderCreatedCounter = Counter.builder("orders.created")
                .tag("service", "order-service")
                .description("Total number of orders created")
                .register(meterRegistry);

        // Timer: measures duration + count
        this.orderProcessingTimer = io.micrometer.core.instrument.Timer.builder("orders.processing.time")
                .tag("service", "order-service")
                .description("Time taken to process an order")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        // Gauge: current value snapshot
        this.pendingOrdersGauge = io.micrometer.core.instrument.Gauge.builder(
                "orders.pending", pendingOrderCount, java.util.concurrent.atomic.AtomicInteger::get)
                .description("Current number of pending orders")
                .register(meterRegistry);
    }

    public void recordOrderCreated(String region) {
        orderCreatedCounter.increment();
        meterRegistry.counter("orders.created.by.region", "region", region).increment();
    }

    public void recordOrderProcessingTime(Runnable orderProcessing) {
        orderProcessingTimer.record(orderProcessing);
    }

    public void setPendingOrders(int count) {
        pendingOrderCount.set(count);
    }
}

// Import for Counter builder
import io.micrometer.core.instrument.Counter;

// --- 8c. Distributed Tracing with Micrometer ---
@org.springframework.stereotype.Service
class TracedOrderService {

    private final io.micrometer.tracing.Tracer tracer;

    public TracedOrderService(io.micrometer.tracing.Tracer tracer) {
        this.tracer = tracer;
    }

    public void processOrderWithCustomSpan(String orderId) {
        // Create a custom span for a logical unit of work
        var span = tracer.nextSpan()
                .name("process-order-payment")
                .tag("order.id", orderId)
                .tag("service", "order-service");

        try (var ws = tracer.withSpan(span.start())) {
            // All logs within this scope will have traceId + spanId
            // Downstream HTTP calls will propagate the trace context automatically
            performPaymentProcessing(orderId);
            span.tag("result", "success");
        } catch (Exception e) {
            span.error(e);
            throw e;
        } finally {
            span.end();
        }
    }

    private void performPaymentProcessing(String orderId) { /* ... */ }
}

// Actuator config (application.yml):
// management:
//   endpoints:
//     web:
//       exposure:
//         include: health,info,metrics,prometheus,loggers,circuitbreakers,env
//   endpoint:
//     health:
//       show-details: when-authorized
//       probes:
//         enabled: true    ← enables /health/liveness and /health/readiness
//   tracing:
//     sampling:
//       probability: 1.0   ← 100% sampling (use 0.1 in production)
//   zipkin:
//     tracing:
//       endpoint: http://zipkin:9411/api/v2/spans

// ============================================================================
// SECTION 9: SECURITY — JWT, OAuth2, Spring Security, mTLS
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: How does JWT-based authentication work in microservices?
 * A: 1. Client authenticates with Auth Service → receives JWT (signed with private key).
 *    2. Client sends JWT in Authorization: Bearer <token> header.
 *    3. Each microservice validates JWT signature (using public key — no DB lookup needed).
 *    4. JWT contains claims: sub (user ID), roles, exp (expiry), iat (issued at).
 *    Stateless — no session, no shared DB. But: can't revoke before expiry (use short expiry + refresh tokens).
 *
 * Q: How do you implement OAuth2 Resource Server in Spring Boot?
 * A: Add spring-boot-starter-oauth2-resource-server. Configure jwk-set-uri (Auth0/Keycloak URL).
 *    Spring validates the token on every request. Use @PreAuthorize for method security.
 *
 * Q: What is the difference between Authentication and Authorization?
 * A: Authentication: WHO are you? (verify identity via JWT/session/API key)
 *    Authorization: WHAT can you do? (verify permissions via roles/scopes)
 *
 * Q: How do you secure service-to-service communication?
 * A: Option 1: Forward the user's JWT from the original request (Feign interceptor).
 *    Option 2: Service-to-service tokens (client credentials OAuth2 flow — no user involved).
 *    Option 3: mTLS (mutual TLS) — each service has a certificate; verifies each other.
 *    Best: mTLS + JWT together (mTLS for transport identity, JWT for user context).
 *
 * Q: @PreAuthorize vs @Secured vs @RolesAllowed?
 * A: @PreAuthorize: most powerful — supports SpEL (hasRole, hasAuthority, #param checks).
 *    @Secured: Spring-specific, role names only.
 *    @RolesAllowed: JSR-250, role names only.
 *    Use @PreAuthorize — it's the most flexible.
 */

// --- 9a. Security Configuration (Spring Boot 3.x / Security 6.x) ---
@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true) // Enables @PreAuthorize
class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        return http
            // Disable CSRF for stateless REST APIs (JWT-based, no cookies)
            .csrf(csrf -> csrf.disable())

            // Stateless session — no HttpSession created
            .sessionManagement(sm -> sm
                .sessionCreationPolicy(
                    org.springframework.security.config.http.SessionCreationPolicy.STATELESS))

            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/actuator/health/**", "/actuator/info").permitAll()
                .requestMatchers("/api/v1/auth/**").permitAll()
                .requestMatchers(HttpMethod.GET, "/api/v1/products/**").hasAnyRole("USER", "ADMIN")
                .requestMatchers("/api/v1/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )

            // OAuth2 Resource Server — validates JWT on every request
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt(jwt -> jwt
                    .jwtAuthenticationConverter(jwtAuthenticationConverter())))

            // Custom 401/403 handlers
            .exceptionHandling(ex -> ex
                .authenticationEntryPoint((req, res, e) -> {
                    res.setStatus(HttpStatus.UNAUTHORIZED.value());
                    res.setContentType(MediaType.APPLICATION_JSON_VALUE);
                    res.getWriter().write("{\"error\":\"Unauthorized\",\"message\":\"" + e.getMessage() + "\"}");
                })
                .accessDeniedHandler((req, res, e) -> {
                    res.setStatus(HttpStatus.FORBIDDEN.value());
                    res.getWriter().write("{\"error\":\"Forbidden\"}");
                })
            )
            .build();
    }

    // Extract roles from JWT claims and map to Spring Security GrantedAuthorities
    @Bean
    public org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationConverter
            jwtAuthenticationConverter() {

        var rolesConverter = new org.springframework.security.oauth2.server.resource
                .authentication.JwtGrantedAuthoritiesConverter();
        rolesConverter.setAuthoritiesClaimName("roles");    // custom claim name
        rolesConverter.setAuthorityPrefix("ROLE_");        // prefix required for hasRole()

        var converter = new org.springframework.security.oauth2.server.resource
                .authentication.JwtAuthenticationConverter();
        converter.setJwtGrantedAuthoritiesConverter(rolesConverter);
        return converter;
    }

    // JWK Set URI configured in application.yml:
    // spring.security.oauth2.resourceserver.jwt.jwk-set-uri=http://auth-service/oauth2/jwks
    @Bean
    public JwtDecoder jwtDecoder(
            org.springframework.boot.autoconfigure.security.oauth2.resource.servlet.OAuth2ResourceServerProperties properties) {
        return org.springframework.security.oauth2.jwt.NimbusJwtDecoder
                .withJwkSetUri(properties.getJwt().getJwkSetUri())
                .build();
    }
}

// --- 9b. Method-Level Security ---
@org.springframework.stereotype.Service
class SecuredOrderService {

    // Only admin or the order owner can access
    @PreAuthorize("hasRole('ADMIN') or #customerId == authentication.name")
    public List<OrderResponse> getOrdersByCustomer(String customerId) {
        return Collections.emptyList();
    }

    // Only users with SCOPE_orders:write (OAuth2 scope)
    @PreAuthorize("hasAuthority('SCOPE_orders:write')")
    public OrderResponse createSecuredOrder(CreateOrderRequest request) {
        return null;
    }

    // Post-filter: returns only items belonging to the current user
    @PreAuthorize("isAuthenticated()")
    @PostFilter("filterObject.customerId == authentication.name")
    public List<OrderResponse> listAllOrders() {
        return Collections.emptyList(); // filtered after method returns
    }
}

// ============================================================================
// SECTION 10: EVENT-DRIVEN ARCHITECTURE — Kafka, Spring Events, Outbox Pattern
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: Why use Kafka over REST for inter-service communication?
 * A: Kafka: decoupled, async, durable, replayable, fan-out (multiple consumers), backpressure.
 *    REST: synchronous coupling, no built-in replay, one-to-one.
 *    Use Kafka for: order-placed events, notifications, audit logs, data pipelines.
 *    Use REST for: immediate query/response (get product details, check availability).
 *
 * Q: What is a Consumer Group in Kafka?
 * A: A group of consumers that collectively consume all partitions of a topic.
 *    Each partition is assigned to exactly ONE consumer in the group at a time.
 *    Multiple groups on the same topic = fan-out (each group gets all messages independently).
 *    Scaling: add more consumers up to the number of partitions.
 *
 * Q: How do you ensure exactly-once semantics in Kafka?
 * A: Producer: enable.idempotence=true + transactional.id → at-least-once deduped to exactly-once.
 *    Consumer: read_committed isolation level.
 *    Spring: @Transactional + KafkaTransactionManager (or ChainedKafkaTransactionManager with DB).
 *
 * Q: What is the Outbox Pattern and why is it needed?
 * A: Dual-write problem: saving to DB and publishing to Kafka in the same operation is not atomic.
 *    If DB saves but Kafka publish fails (or vice versa), data is inconsistent.
 *    Outbox: save event to an OUTBOX table in the SAME DB transaction as business data.
 *    A separate process (Debezium CDC or polling publisher) reads the outbox and publishes to Kafka.
 *    Guarantees at-least-once delivery with exactly the same transaction semantics as the DB write.
 *
 * Q: What is @KafkaListener and how do you handle failures?
 * A: @KafkaListener annotates a method as a Kafka consumer.
 *    Failure handling: DefaultErrorHandler with BackOff + DeadLetterPublishingRecoverer.
 *    Retries N times with exponential backoff; after exhaustion, sends to a .DLT topic.
 */

// --- 10a. Domain Events ---
record OrderPlacedEvent(
    String orderId,
    String customerId,
    BigDecimal totalAmount,
    LocalDateTime occurredAt,
    String traceId
) {}

record OrderCancelledEvent(
    String orderId,
    String customerId,
    String reason,
    LocalDateTime occurredAt
) {}

record InventoryReservedEvent(
    String reservationId,
    String orderId,
    String productId,
    int quantity
) {}

// --- 10b. Kafka Producer Configuration ---
@Configuration
class KafkaProducerConfig {

    @Bean
    public ProducerFactory<String, Object> producerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(org.apache.kafka.clients.producer.ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(org.apache.kafka.clients.producer.ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
                org.apache.kafka.common.serialization.StringSerializer.class);
        props.put(org.apache.kafka.clients.producer.ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
                org.springframework.kafka.support.serializer.JsonSerializer.class);

        // Idempotent producer — prevents duplicate messages on retry
        props.put(org.apache.kafka.clients.producer.ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        props.put(org.apache.kafka.clients.producer.ProducerConfig.ACKS_CONFIG, "all"); // all ISR acks
        props.put(org.apache.kafka.clients.producer.ProducerConfig.RETRIES_CONFIG, Integer.MAX_VALUE);
        props.put(org.apache.kafka.clients.producer.ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, 5);

        return new org.springframework.kafka.core.DefaultKafkaProducerFactory<>(props);
    }

    @Bean
    public KafkaTemplate<String, Object> kafkaTemplate(ProducerFactory<String, Object> pf) {
        KafkaTemplate<String, Object> template = new KafkaTemplate<>(pf);
        template.setObservationEnabled(true); // Micrometer tracing integration
        return template;
    }
}

// --- 10c. Event Publisher Service ---
@org.springframework.stereotype.Service
class OrderEventPublisher {

    private static final String ORDERS_TOPIC = "orders.events";
    private static final String ORDER_PLACED_TOPIC = "orders.placed";

    private final KafkaTemplate<String, Object> kafkaTemplate;

    public OrderEventPublisher(KafkaTemplate<String, Object> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    @Transactional // Ensures Kafka send is part of DB transaction (with ChainedTransactionManager)
    public void publishOrderPlaced(OrderPlacedEvent event) {
        // Use orderId as key → same order's events go to same partition → ordered processing
        var record = new org.apache.kafka.clients.producer.ProducerRecord<String, Object>(
                ORDER_PLACED_TOPIC, event.orderId(), event);

        // Add headers for tracing and event type
        record.headers()
                .add("event-type", "OrderPlaced".getBytes())
                .add("source-service", "order-service".getBytes())
                .add("content-type", "application/json".getBytes());

        kafkaTemplate.send(record)
                .whenComplete((result, ex) -> {
                    if (ex != null) {
                        System.err.println("Failed to publish OrderPlacedEvent: " + ex.getMessage());
                        // In production: save to outbox table for retry
                    } else {
                        System.out.printf("Published OrderPlaced [%s] to partition %d, offset %d%n",
                                event.orderId(),
                                result.getRecordMetadata().partition(),
                                result.getRecordMetadata().offset());
                    }
                });
    }
}

// --- 10d. Kafka Consumer Configuration ---
@Configuration
class KafkaConsumerConfig {

    @Bean
    public ConsumerFactory<String, Object> consumerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.GROUP_ID_CONFIG, "inventory-service-group");
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG,
                org.apache.kafka.common.serialization.StringDeserializer.class);
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG,
                org.springframework.kafka.support.serializer.JsonDeserializer.class);
        props.put(org.springframework.kafka.support.serializer.JsonDeserializer.TRUSTED_PACKAGES, "interview.springboot");
        // Manual commit for fine-grained control
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);
        return new org.springframework.kafka.core.DefaultKafkaConsumerFactory<>(props);
    }

    @Bean
    public org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory<String, Object>
            kafkaListenerContainerFactory(ConsumerFactory<String, Object> cf) {

        var factory = new org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory<String, Object>();
        factory.setConsumerFactory(cf);
        factory.setConcurrency(3); // 3 threads per listener = 3 partitions consumed in parallel
        factory.getContainerProperties().setAckMode(
                org.springframework.kafka.listener.ContainerProperties.AckMode.MANUAL_IMMEDIATE);
        factory.setCommonErrorHandler(errorHandler());
        factory.setObservationEnabled(true); // distributed tracing
        return factory;
    }

    @Bean
    public org.springframework.kafka.listener.DefaultErrorHandler errorHandler() {
        // Exponential backoff: 1s, 2s, 4s, then DLT
        var backoff = new org.springframework.util.backoff.ExponentialBackOff(1000L, 2.0);
        backoff.setMaxElapsedTime(10000L); // max 10 seconds
        backoff.setMaxInterval(4000L);

        var dltPublisher = new org.springframework.kafka.listener.DeadLetterPublishingRecoverer(
                new KafkaTemplate<>(new org.springframework.kafka.core.DefaultKafkaProducerFactory<>(new HashMap<>())));
        // DLT topic naming: orders.placed.DLT (auto-appended)

        return new org.springframework.kafka.listener.DefaultErrorHandler(dltPublisher, backoff);
    }
}

// --- 10e. Kafka Consumer ---
@org.springframework.stereotype.Component
class InventoryEventListener {

    private final ResilientInventoryService inventoryService;

    public InventoryEventListener(ResilientInventoryService inventoryService) {
        this.inventoryService = inventoryService;
    }

    @KafkaListener(
        topics = "orders.placed",
        groupId = "inventory-service-group",
        containerFactory = "kafkaListenerContainerFactory"
    )
    @Transactional // DB operation + offset commit in same transaction
    public void handleOrderPlaced(
            OrderPlacedEvent event,
            org.springframework.kafka.support.Acknowledgment ack,
            @org.springframework.messaging.handler.annotation.Header(org.springframework.kafka.support.KafkaHeaders.RECEIVED_PARTITION) int partition,
            @org.springframework.messaging.handler.annotation.Header(org.springframework.kafka.support.KafkaHeaders.OFFSET) long offset) {

        System.out.printf("Processing OrderPlaced [%s] from partition=%d offset=%d%n",
                event.orderId(), partition, offset);

        try {
            // Idempotent processing — check if already processed
            if (alreadyProcessed(event.orderId())) {
                ack.acknowledge(); // Commit offset, skip processing
                return;
            }

            // Reserve inventory
            inventoryService.checkInventorySafely(event.orderId());

            // Mark as processed (idempotency key in DB)
            markAsProcessed(event.orderId());

            ack.acknowledge(); // Manual commit ONLY after successful processing
        } catch (Exception e) {
            System.err.println("Failed to process OrderPlaced: " + e.getMessage());
            // Don't acknowledge → offset NOT committed → message redelivered
            // ErrorHandler will retry with backoff, then send to DLT
            throw e;
        }
    }

    // Batch listener example — higher throughput
    @KafkaListener(topics = "analytics.events", groupId = "analytics-group")
    public void handleBatch(
            List<org.springframework.messaging.Message<OrderPlacedEvent>> messages,
            org.springframework.kafka.support.Acknowledgment ack) {

        System.out.println("Processing batch of " + messages.size() + " events");
        // Process all, then commit
        ack.acknowledge();
    }

    private boolean alreadyProcessed(String orderId) { return false; } // check idempotency table
    private void markAsProcessed(String orderId) { /* insert into processed_events */ }
}

// --- 10f. Outbox Pattern Implementation ---
@Entity
@Table(name = "outbox_events")
class OutboxEvent {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    @Column(nullable = false)
    private String aggregateType; // "Order"

    @Column(nullable = false)
    private String aggregateId;   // orderId

    @Column(nullable = false)
    private String eventType;     // "OrderPlaced"

    @Column(columnDefinition = "TEXT", nullable = false)
    private String payload;       // JSON-serialized event

    @Enumerated(EnumType.STRING)
    private OutboxStatus status = OutboxStatus.PENDING;

    private LocalDateTime createdAt = LocalDateTime.now();
    private LocalDateTime processedAt;
    private int retryCount = 0;

    // Getters/setters
    public String getId() { return id; }
    public String getAggregateId() { return aggregateId; }
    public String getEventType() { return eventType; }
    public String getPayload() { return payload; }
    public OutboxStatus getStatus() { return status; }
    public void setStatus(OutboxStatus status) { this.status = status; }
    public void setProcessedAt(LocalDateTime t) { this.processedAt = t; }
    public void setAggregateType(String t) { this.aggregateType = t; }
    public void setAggregateId(String id) { this.aggregateId = id; }
    public void setEventType(String t) { this.eventType = t; }
    public void setPayload(String p) { this.payload = p; }
    public int getRetryCount() { return retryCount; }
    public void incrementRetry() { this.retryCount++; }
}

enum OutboxStatus { PENDING, PUBLISHED, FAILED }

@org.springframework.stereotype.Repository
interface OutboxRepository extends JpaRepository<OutboxEvent, String> {
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT o FROM OutboxEvent o WHERE o.status = 'PENDING' ORDER BY o.createdAt ASC")
    List<OutboxEvent> findPendingEventsForProcessing(org.springframework.data.domain.Pageable pageable);
}

// Outbox Publisher — runs on schedule, polls outbox table and publishes to Kafka
@org.springframework.stereotype.Component
class OutboxPublisher {

    private final OutboxRepository outboxRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    public OutboxPublisher(OutboxRepository outboxRepository,
                           KafkaTemplate<String, Object> kafkaTemplate,
                           com.fasterxml.jackson.databind.ObjectMapper objectMapper) {
        this.outboxRepository = outboxRepository;
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
    }

    @org.springframework.scheduling.annotation.Scheduled(fixedDelay = 5000) // every 5 seconds
    @Transactional
    public void publishPendingEvents() {
        var pending = outboxRepository.findPendingEventsForProcessing(
                org.springframework.data.domain.PageRequest.of(0, 100));

        for (OutboxEvent event : pending) {
            try {
                String topic = resolveTopicForEvent(event.getEventType());
                kafkaTemplate.send(topic, event.getAggregateId(), event.getPayload()).get(); // sync for reliability
                event.setStatus(OutboxStatus.PUBLISHED);
                event.setProcessedAt(LocalDateTime.now());
            } catch (Exception e) {
                event.incrementRetry();
                if (event.getRetryCount() >= 5) {
                    event.setStatus(OutboxStatus.FAILED);
                }
            }
        }
    }

    private String resolveTopicForEvent(String eventType) {
        return switch (eventType) {
            case "OrderPlaced" -> "orders.placed";
            case "OrderCancelled" -> "orders.cancelled";
            default -> "events.misc";
        };
    }
}


// ============================================================================
// SECTION 11: SAGA PATTERN & DISTRIBUTED TRANSACTIONS
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: Why can't you use a traditional 2-Phase Commit (2PC) in microservices?
 * A: 2PC requires a coordinator that holds locks across all participants until all commit.
 *    In microservices: blocking locks across network-partitioned services → severe performance issues,
 *    single point of failure (coordinator), incompatible with polyglot persistence.
 *    Use Saga instead — eventual consistency without distributed locks.
 *
 * Q: What is the Saga pattern and what are its two implementations?
 * A: Saga: sequence of local transactions, each publishing an event/message.
 *    If any step fails, compensating transactions are executed in reverse order.
 *
 *    Choreography: services react to each other's events — no central coordinator.
 *      Pros: simple, decoupled. Cons: hard to track flow, risk of cyclic dependencies.
 *
 *    Orchestration: a central Saga Orchestrator tells each service what to do.
 *      Pros: clear flow, easier to monitor. Cons: orchestrator is a central component.
 *      Better for complex workflows with many steps.
 *
 * Q: What is a compensating transaction?
 * A: The undo operation for a completed Saga step. Example: if payment succeeds but
 *    inventory reservation fails, the compensating transaction for payment is a refund.
 *    Compensating transactions must be idempotent (may be retried).
 *
 * Q: What is the difference between ACID and BASE?
 * A: ACID (single DB): Atomicity, Consistency, Isolation, Durability — strong guarantees.
 *    BASE (microservices): Basically Available, Soft state, Eventually consistent.
 *    Sagas are BASE — eventual consistency, not immediate.
 */

// --- 11a. Choreography-based Saga ---
// Flow: OrderService → (OrderPlaced) → InventoryService → (InventoryReserved) → PaymentService
//                                                        → (InventoryFailed) → OrderService.cancel()

@org.springframework.stereotype.Service
class OrderSagaChoreographyService {

    private final OrderRepository orderRepository;
    private final OutboxPublisher outboxPublisher;
    private final OutboxRepository outboxRepo;
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    public OrderSagaChoreographyService(OrderRepository orderRepository,
                                         OutboxPublisher outboxPublisher,
                                         OutboxRepository outboxRepo,
                                         com.fasterxml.jackson.databind.ObjectMapper objectMapper) {
        this.orderRepository = orderRepository;
        this.outboxPublisher = outboxPublisher;
        this.outboxRepo = outboxRepo;
        this.objectMapper = objectMapper;
    }

    // Step 1: Create order and publish event (atomic via Outbox)
    @Transactional
    public OrderResponse initiateOrderSaga(CreateOrderRequest request) throws Exception {
        // Create order in PENDING state
        Order order = new Order();
        order.setCustomerId(request.customerId());
        order.setStatus(OrderStatus.PENDING);
        order = orderRepository.save(order);

        // Write to outbox IN THE SAME TRANSACTION — atomic
        OutboxEvent outboxEvent = new OutboxEvent();
        outboxEvent.setAggregateType("Order");
        outboxEvent.setAggregateId(order.getId());
        outboxEvent.setEventType("OrderPlaced");
        outboxEvent.setPayload(objectMapper.writeValueAsString(
                new OrderPlacedEvent(order.getId(), order.getCustomerId(),
                        order.getTotalAmount(), LocalDateTime.now(), "")));
        outboxRepo.save(outboxEvent);

        // Outbox publisher will async publish to Kafka → triggers next saga step
        return new OrderResponse(order.getId(), order.getCustomerId(),
                order.getStatus().name(), order.getTotalAmount(), LocalDateTime.now());
    }

    // Compensation handler — called when downstream step fails
    @KafkaListener(topics = "inventory.reservation.failed", groupId = "order-saga-group")
    @Transactional
    public void handleInventoryFailure(InventoryFailedEvent event,
                                        org.springframework.kafka.support.Acknowledgment ack) {
        orderRepository.findById(event.orderId()).ifPresent(order -> {
            order.cancel(); // Compensate: cancel the order
            orderRepository.save(order);
            // Publish OrderCancelled event for further downstream compensation
        });
        ack.acknowledge();
    }
}

record InventoryFailedEvent(String orderId, String reason) {}

// --- 11b. Orchestration-based Saga ---
// Central orchestrator coordinates all steps

enum SagaStatus { STARTED, INVENTORY_RESERVED, PAYMENT_PROCESSED, COMPLETED,
                  COMPENSATING_INVENTORY, COMPENSATING_PAYMENT, FAILED }

@Entity
@Table(name = "order_sagas")
class OrderSagaState {
    @Id private String sagaId;
    private String orderId;
    private String customerId;
    private BigDecimal amount;

    @Enumerated(EnumType.STRING)
    private SagaStatus status;

    private String failureReason;
    private LocalDateTime startedAt;
    private LocalDateTime completedAt;

    public String getSagaId() { return sagaId; }
    public void setSagaId(String id) { this.sagaId = id; }
    public String getOrderId() { return orderId; }
    public void setOrderId(String id) { this.orderId = id; }
    public SagaStatus getStatus() { return status; }
    public void setStatus(SagaStatus s) { this.status = s; }
    public String getCustomerId() { return customerId; }
    public void setCustomerId(String id) { this.customerId = id; }
    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal a) { this.amount = a; }
    public void setStartedAt(LocalDateTime t) { this.startedAt = t; }
    public void setFailureReason(String r) { this.failureReason = r; }
    public void setCompletedAt(LocalDateTime t) { this.completedAt = t; }
}

@org.springframework.stereotype.Service
class OrderSagaOrchestrator {

    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final JpaRepository<OrderSagaState, String> sagaRepository;

    @SuppressWarnings("unchecked")
    public OrderSagaOrchestrator(KafkaTemplate<String, Object> kafkaTemplate,
                                  JpaRepository sagaRepository) {
        this.kafkaTemplate = kafkaTemplate;
        this.sagaRepository = sagaRepository;
    }

    @Transactional
    public void startSaga(String orderId, String customerId, BigDecimal amount) {
        OrderSagaState saga = new OrderSagaState();
        saga.setSagaId(UUID.randomUUID().toString());
        saga.setOrderId(orderId);
        saga.setCustomerId(customerId);
        saga.setAmount(amount);
        saga.setStatus(SagaStatus.STARTED);
        saga.setStartedAt(LocalDateTime.now());
        sagaRepository.save(saga);

        // Step 1: Send command to inventory service
        kafkaTemplate.send("inventory.commands", orderId,
                new ReserveStockRequest(null, orderId, 1));
    }

    // Called when inventory says OK
    @KafkaListener(topics = "inventory.reserved", groupId = "saga-orchestrator")
    @Transactional
    public void onInventoryReserved(InventoryReservedEvent event,
                                     org.springframework.kafka.support.Acknowledgment ack) {
        sagaRepository.findAll().stream()
                .filter(s -> s.getOrderId().equals(event.orderId()))
                .findFirst()
                .ifPresent(saga -> {
                    saga.setStatus(SagaStatus.INVENTORY_RESERVED);
                    sagaRepository.save(saga);
                    // Step 2: Send command to payment service
                    kafkaTemplate.send("payment.commands", saga.getOrderId(),
                            new PaymentRequest(saga.getOrderId(), saga.getAmount(), "INR"));
                });
        ack.acknowledge();
    }

    // Called when any step fails — start compensation
    @KafkaListener(topics = "inventory.reservation.failed", groupId = "saga-orchestrator")
    @Transactional
    public void onInventoryFailed(InventoryFailedEvent event,
                                   org.springframework.kafka.support.Acknowledgment ack) {
        sagaRepository.findAll().stream()
                .filter(s -> s.getOrderId().equals(event.orderId()))
                .findFirst()
                .ifPresent(saga -> {
                    saga.setStatus(SagaStatus.FAILED);
                    saga.setFailureReason(event.reason());
                    sagaRepository.save(saga);
                    // Compensation: publish OrderCancelled command
                    kafkaTemplate.send("order.commands.cancel", saga.getOrderId(), saga.getOrderId());
                });
        ack.acknowledge();
    }
}

// ============================================================================
// SECTION 12: CQRS & EVENT SOURCING
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: What is CQRS and what problem does it solve?
 * A: Command Query Responsibility Segregation — separates the Write Model (Commands)
 *    from the Read Model (Queries).
 *    Problem: a single model optimized for writes (normalized, transactional) is often poor
 *    for reads (complex joins, aggregations). CQRS lets you optimize each independently.
 *    Write side: normalized DB with business rules.
 *    Read side: denormalized views (flat tables, Elasticsearch, Redis) optimized for queries.
 *
 * Q: Does CQRS require Event Sourcing?
 * A: NO — they complement each other but are independent.
 *    CQRS can use two different SQL tables for read/write.
 *    Event Sourcing uses an event log as the primary store.
 *    Together: write side stores events; read side is a projection of those events.
 *
 * Q: What is Event Sourcing?
 * A: Instead of storing current state, store ALL events that led to that state.
 *    Current state = replay of all events in order.
 *    Benefits: full audit log, time travel (reconstruct state at any point), event replay.
 *    Drawbacks: complex, eventual consistency for read models, event schema evolution.
 *
 * Q: What is a Projection in Event Sourcing?
 * A: A read model built by listening to events and updating a denormalized view.
 *    Multiple projections can exist for the same event stream (order history, analytics, search index).
 */

// --- 12a. Command / Query separation ---
// Commands (write side) — represent intent to change state
record PlaceOrderCommand(String customerId, List<OrderItemRequest> items, LocalDate deliveryDate) {}
record CancelOrderCommand(String orderId, String reason) {}
record ConfirmOrderCommand(String orderId) {}

// Queries (read side) — no state change, different model
record OrderQuery(String orderId) {}
record CustomerOrdersQuery(String customerId, int page, int size) {}

// --- 12b. Command Handler (write side) ---
@org.springframework.stereotype.Service
class OrderCommandHandler {

    private final OrderRepository orderRepository;
    private final OrderEventPublisher eventPublisher;

    public OrderCommandHandler(OrderRepository orderRepository, OrderEventPublisher eventPublisher) {
        this.orderRepository = orderRepository;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public String handle(PlaceOrderCommand cmd) {
        Order order = new Order();
        order.setCustomerId(cmd.customerId());
        order.setStatus(OrderStatus.PENDING);
        order = orderRepository.save(order);
        // Publish domain event
        eventPublisher.publishOrderPlaced(new OrderPlacedEvent(
                order.getId(), order.getCustomerId(), order.getTotalAmount(),
                LocalDateTime.now(), ""));
        return order.getId();
    }

    @Transactional
    public void handle(CancelOrderCommand cmd) {
        Order order = orderRepository.findById(cmd.orderId())
                .orElseThrow(() -> new ResourceNotFoundException("Order not found: " + cmd.orderId()));
        order.cancel();
        orderRepository.save(order);
    }
}

// --- 12c. Read Model (denormalized for fast queries) ---
@Entity
@Table(name = "order_read_model")
class OrderReadModel {
    @Id private String orderId;
    private String customerId;
    private String customerName;  // denormalized from Customer service
    private String customerEmail; // denormalized
    private String status;
    private BigDecimal totalAmount;
    private int itemCount;
    private String primaryProductName; // denormalized
    private LocalDateTime createdAt;
    private LocalDateTime lastUpdated;

    // All getters and setters
    public String getOrderId() { return orderId; }
    public void setOrderId(String id) { this.orderId = id; }
    public void setCustomerId(String id) { this.customerId = id; }
    public void setStatus(String s) { this.status = s; }
    public void setTotalAmount(BigDecimal a) { this.totalAmount = a; }
    public void setCreatedAt(LocalDateTime t) { this.createdAt = t; }
    public void setLastUpdated(LocalDateTime t) { this.lastUpdated = t; }
    public void setItemCount(int c) { this.itemCount = c; }
}

@org.springframework.stereotype.Repository
interface OrderReadModelRepository extends JpaRepository<OrderReadModel, String> {
    List<OrderReadModel> findByCustomerIdOrderByCreatedAtDesc(String customerId);

    @Query("SELECT o FROM OrderReadModel o WHERE o.status = :status AND o.createdAt >= :since")
    List<OrderReadModel> findByStatusSince(@Param("status") String status, @Param("since") LocalDateTime since);
}

// --- 12d. Projection Builder — listens to events, updates read model ---
@org.springframework.stereotype.Component
class OrderProjection {

    private final OrderReadModelRepository readModelRepository;

    public OrderProjection(OrderReadModelRepository readModelRepository) {
        this.readModelRepository = readModelRepository;
    }

    @KafkaListener(topics = {"orders.placed", "orders.cancelled", "orders.confirmed"},
            groupId = "order-projection-group")
    @Transactional
    public void onOrderEvent(
            org.springframework.messaging.Message<String> message,
            org.springframework.kafka.support.Acknowledgment ack) {

        String eventType = new String(
                message.getHeaders().get("event-type", byte[].class));

        // Update the read model based on event type
        switch (eventType) {
            case "OrderPlaced" -> {
                // Parse and create read model entry
                OrderReadModel rm = new OrderReadModel();
                rm.setOrderId(UUID.randomUUID().toString());
                rm.setStatus("PENDING");
                rm.setCreatedAt(LocalDateTime.now());
                rm.setLastUpdated(LocalDateTime.now());
                readModelRepository.save(rm);
            }
            case "OrderCancelled" -> {
                // Update status in read model
                readModelRepository.findAll()
                        .stream().findFirst()
                        .ifPresent(rm -> {
                            rm.setStatus("CANCELLED");
                            rm.setLastUpdated(LocalDateTime.now());
                            readModelRepository.save(rm);
                        });
            }
        }
        ack.acknowledge();
    }
}

// ============================================================================
// SECTION 13: CONFIGURATION MANAGEMENT
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: How does Spring Cloud Config Server work?
 * A: Centralized config server backed by Git, filesystem, or Vault.
 *    Microservices fetch config on startup from: /{application}/{profile}/{label}
 *    URL example: http://config-server/order-service/production/main
 *    Returns merged properties: default + application-specific + profile-specific.
 *
 * Q: How do you refresh configuration without restarting a service?
 * A: Add @RefreshScope to beans that should re-initialize on refresh.
 *    Trigger: POST /actuator/refresh (single instance) or Spring Cloud Bus (all instances via Kafka/RabbitMQ).
 *
 * Q: How do you manage secrets (passwords, API keys)?
 * A: NEVER store in Git (even private). Options:
 *    1. HashiCorp Vault (spring-vault-core) — dynamic secrets, rotation, audit log.
 *    2. AWS Secrets Manager / Parameter Store — cloud-native, auto-rotation.
 *    3. Kubernetes Secrets (base64, not encrypted — use Sealed Secrets or External Secrets Operator).
 *    4. Encrypt in Spring Cloud Config with symmetric/asymmetric key.
 *
 * Q: Spring Profiles best practices?
 * A: Profiles: local, dev, staging, prod. Activate via:
 *    SPRING_PROFILES_ACTIVE=prod (env var — highest priority)
 *    --spring.profiles.active=prod (cmd line arg)
 *    spring.profiles.active in application.properties (lowest)
 *    Never check prod credentials into VCS.
 */

// --- 13a. Config Server Setup ---
// Server application:
// @SpringBootApplication @EnableConfigServer
// class ConfigServerApp { ... }
//
// application.yml on server:
// spring.cloud.config.server.git.uri=https://github.com/org/config-repo
// spring.cloud.config.server.git.search-paths=microservices/{application}

// --- 13b. RefreshScope Bean ---
@Configuration
class RefreshableDemoConfig {

    // @RefreshScope: bean is destroyed and re-created on /actuator/refresh
    // Use for beans that read @Value or @ConfigurationProperties that can change at runtime
    @Bean
    @org.springframework.cloud.context.config.annotation.RefreshScope
    public FeatureFlagService featureFlagService(
            @org.springframework.beans.factory.annotation.Value("${features.new-checkout:false}")
            boolean newCheckoutEnabled) {
        return new FeatureFlagService(newCheckoutEnabled);
    }
}

class FeatureFlagService {
    private final boolean newCheckoutEnabled;
    public FeatureFlagService(boolean newCheckoutEnabled) {
        this.newCheckoutEnabled = newCheckoutEnabled;
    }
    public boolean isNewCheckoutEnabled() { return newCheckoutEnabled; }
}

// --- 13c. Vault Integration ---
// application.yml:
// spring:
//   cloud:
//     vault:
//       host: vault.prod.internal
//       port: 8200
//       scheme: https
//       authentication: KUBERNETES   ← K8s service account token
//       kubernetes:
//         role: order-service
//       kv:
//         enabled: true
//         backend: secret
//         default-context: order-service
// Vault secrets at secret/order-service are automatically injected as Spring properties

// ============================================================================
// SECTION 14: CONTAINERIZATION & KUBERNETES INTEGRATION
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: How do you build a Docker image for a Spring Boot app?
 * A: Option 1: spring-boot-maven-plugin -> mvn spring-boot:build-image (uses Buildpacks, no Dockerfile).
 *    Option 2: Dockerfile with multi-stage build + layered JARs.
 *    Spring Boot 3.x supports layered JARs: java -Djarmode=layertools -jar app.jar extract
 *    Layers: dependencies, spring-boot-loader, snapshot-dependencies, application.
 *    Only the 'application' layer changes on code changes → faster builds, smaller image diffs.
 *
 * Q: What Kubernetes probes does Spring Boot Actuator expose?
 * A: /actuator/health/liveness  → LivenessState (CORRECT = alive, BROKEN = restart pod)
 *    /actuator/health/readiness → ReadinessState (ACCEPTING_TRAFFIC / REFUSING_TRAFFIC)
 *    Configure in K8s: livenessProbe and readinessProbe pointing to these endpoints.
 *
 * Q: How do you configure Spring Boot for Kubernetes (config, secrets, scaling)?
 * A: Config: ConfigMaps → mounted as files or env vars → Spring reads via spring.config.location.
 *    Secrets: K8s Secrets → env vars (DB_PASSWORD=...) → referenced in application.yml as ${DB_PASSWORD}.
 *    Spring Cloud Kubernetes: spring-cloud-kubernetes-client-config reads ConfigMaps directly via API.
 *    Scaling: HPA (Horizontal Pod Autoscaler) + stateless services + externalized sessions (Redis).
 *
 * Q: What is graceful shutdown and how do you configure it?
 * A: On SIGTERM: stop accepting new requests, finish in-flight requests, then shut down.
 *    Spring Boot: server.shutdown=graceful + spring.lifecycle.timeout-per-shutdown-phase=30s
 *    K8s: set terminationGracePeriodSeconds=60 in pod spec.
 *    Kafka: @KafkaListener containers are stopped before the context is closed.
 */

// --- 14a. Dockerfile (multi-stage with layered JARs) ---
// # Stage 1: Extract layers
// FROM eclipse-temurin:21-jre AS builder
// WORKDIR /app
// COPY target/app.jar app.jar
// RUN java -Djarmode=layertools -jar app.jar extract
//
// # Stage 2: Final image
// FROM eclipse-temurin:21-jre
// WORKDIR /app
// RUN addgroup --system spring && adduser --system spring --ingroup spring
// USER spring:spring
// COPY --from=builder /app/dependencies/ ./
// COPY --from=builder /app/spring-boot-loader/ ./
// COPY --from=builder /app/snapshot-dependencies/ ./
// COPY --from=builder /app/application/ ./
// ENTRYPOINT ["java", "org.springframework.boot.loader.JarLauncher"]

// --- 14b. Kubernetes Deployment YAML ---
// apiVersion: apps/v1
// kind: Deployment
// metadata:
//   name: order-service
// spec:
//   replicas: 3
//   selector:
//     matchLabels:
//       app: order-service
//   template:
//     spec:
//       containers:
//       - name: order-service
//         image: registry/order-service:1.0.0
//         ports:
//         - containerPort: 8080
//         env:
//         - name: SPRING_PROFILES_ACTIVE
//           value: "prod"
//         - name: DB_PASSWORD
//           valueFrom:
//             secretKeyRef:
//               name: order-service-secrets
//               key: db-password
//         resources:
//           requests: { memory: "256Mi", cpu: "250m" }
//           limits:   { memory: "512Mi", cpu: "500m" }
//         livenessProbe:
//           httpGet: { path: /actuator/health/liveness, port: 8080 }
//           initialDelaySeconds: 30
//           periodSeconds: 10
//         readinessProbe:
//           httpGet: { path: /actuator/health/readiness, port: 8080 }
//           initialDelaySeconds: 10
//           periodSeconds: 5
//         lifecycle:
//           preStop:
//             exec:
//               command: ["sh", "-c", "sleep 10"]  ← wait for LB to deregister

// --- 14c. Graceful Shutdown Configuration ---
// application.yml:
// server:
//   shutdown: graceful
// spring:
//   lifecycle:
//     timeout-per-shutdown-phase: 30s

// In-code: listen to context close event for custom cleanup
@org.springframework.stereotype.Component
class GracefulShutdownHandler {

    private final org.springframework.kafka.config.KafkaListenerEndpointRegistry kafkaListenerRegistry;

    public GracefulShutdownHandler(
            org.springframework.kafka.config.KafkaListenerEndpointRegistry kafkaListenerRegistry) {
        this.kafkaListenerRegistry = kafkaListenerRegistry;
    }

    @org.springframework.context.event.EventListener(org.springframework.context.event.ContextClosedEvent.class)
    public void onContextClose() {
        System.out.println("Context closing — stopping Kafka listeners gracefully");
        kafkaListenerRegistry.getListenerContainers()
                .forEach(container -> container.stop(() ->
                        System.out.println("Listener stopped: " + container.getListenerId())));
    }
}

// ============================================================================
// SECTION 15: TESTING MICROSERVICES
// ============================================================================

/**
 * KEY INTERVIEW QUESTIONS:
 *
 * Q: What are the different levels of testing in microservices?
 * A: Unit Tests: test individual classes/methods in isolation (mock dependencies). Fast.
 *    Integration Tests: test a slice of the app with real components (@DataJpaTest, @WebMvcTest).
 *    Component Tests: test the whole service in isolation (mock external services via WireMock).
 *    Contract Tests: verify service contracts between producer and consumer (Pact, Spring Cloud Contract).
 *    End-to-End Tests: run all services together. Slow, brittle — use sparingly.
 *
 * Q: @SpringBootTest vs @WebMvcTest vs @DataJpaTest?
 * A: @SpringBootTest: loads FULL application context. Use for integration tests. Slow.
 *    @WebMvcTest: loads only web layer (controllers, filters, security). Fast. Mock services.
 *    @DataJpaTest: loads only JPA layer (repositories, DB). Uses H2 by default. Fast.
 *    @MockBean: creates a Mockito mock and adds it to the Spring context.
 *
 * Q: What is WireMock and why is it useful?
 * A: HTTP mock server. Stubs external HTTP dependencies (other microservices) during tests.
 *    Lets you test your service in isolation without running other services.
 *    Supports request/response stubbing, verification, fault simulation (delays, errors).
 *
 * Q: What is Consumer-Driven Contract Testing?
 * A: The CONSUMER defines the contract (what it expects from the provider).
 *    The PROVIDER verifies it meets that contract. Prevents breaking API changes.
 *    Tools: Pact, Spring Cloud Contract. Runs as part of CI/CD pipeline.
 *
 * Q: How do you test Kafka producers and consumers?
 * A: EmbeddedKafka (@EmbeddedKafka) — runs an in-memory Kafka broker in the test JVM.
 *    No external Kafka needed. Fast and isolated.
 */

// --- 15a. Unit Test (Mockito) ---
// @ExtendWith(MockitoExtension.class)
// class OrderServiceTest {
//
//     @Mock OrderRepository orderRepository;
//     @Mock OrderEventPublisher eventPublisher;
//     @InjectMocks OrderService orderService;
//
//     @Test
//     void createOrder_shouldReturnOrderResponse_whenValidRequest() {
//         // Arrange
//         var request = new CreateOrderRequest("cust-1", List.of(), LocalDate.now().plusDays(1));
//         var savedOrder = new Order();
//         // use reflection to set id: ReflectionTestUtils.setField(savedOrder, "id", "order-1");
//         when(orderRepository.save(any())).thenReturn(savedOrder);
//
//         // Act
//         var response = orderService.createOrder(request);
//
//         // Assert
//         assertThat(response).isNotNull();
//         verify(orderRepository, times(1)).save(any(Order.class));
//         verifyNoMoreInteractions(orderRepository);
//     }
//
//     @Test
//     void getOrder_shouldThrowNotFoundException_whenOrderDoesNotExist() {
//         when(orderRepository.findById("missing-id")).thenReturn(Optional.empty());
//         assertThatThrownBy(() -> orderService.getOrder("missing-id"))
//                 .isInstanceOf(ResourceNotFoundException.class)
//                 .hasMessageContaining("missing-id");
//     }
// }

// --- 15b. @WebMvcTest (Controller slice test) ---
// @WebMvcTest(OrderController.class)
// @Import(SecurityConfig.class)
// class OrderControllerTest {
//
//     @Autowired MockMvc mockMvc;
//     @MockBean OrderService orderService;
//     @Autowired ObjectMapper objectMapper;
//
//     @Test
//     @WithMockUser(roles = "USER")
//     void createOrder_shouldReturn201_whenValidRequest() throws Exception {
//         var request = new CreateOrderRequest("cust-1", List.of(), LocalDate.now().plusDays(1));
//         var response = new OrderResponse("order-1", "cust-1", "PENDING", BigDecimal.TEN, LocalDateTime.now());
//         when(orderService.createOrder(any())).thenReturn(response);
//
//         mockMvc.perform(post("/api/v1/orders")
//                 .contentType(MediaType.APPLICATION_JSON)
//                 .content(objectMapper.writeValueAsString(request)))
//                 .andExpect(status().isCreated())
//                 .andExpect(header().exists("Location"))
//                 .andExpect(jsonPath("$.orderId").value("order-1"))
//                 .andExpect(jsonPath("$.status").value("PENDING"));
//     }
//
//     @Test
//     void createOrder_shouldReturn400_whenCustomerIdIsBlank() throws Exception {
//         var invalid = new CreateOrderRequest("", List.of(), null);
//         mockMvc.perform(post("/api/v1/orders")
//                 .contentType(MediaType.APPLICATION_JSON)
//                 .content(objectMapper.writeValueAsString(invalid)))
//                 .andExpect(status().isBadRequest())
//                 .andExpect(jsonPath("$.fieldErrors.customerId").exists());
//     }
// }

// --- 15c. @DataJpaTest (Repository slice test) ---
// @DataJpaTest
// @AutoConfigureTestDatabase(replace = NONE)  // Use real DB (Testcontainers)
// @Testcontainers
// class OrderRepositoryTest {
//
//     @Container
//     static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15")
//             .withDatabaseName("testdb");
//
//     @DynamicPropertySource
//     static void setProps(DynamicPropertyRegistry registry) {
//         registry.add("spring.datasource.url", postgres::getJdbcUrl);
//         registry.add("spring.datasource.username", postgres::getUsername);
//         registry.add("spring.datasource.password", postgres::getPassword);
//     }
//
//     @Autowired OrderRepository orderRepository;
//
//     @Test
//     void findByCustomerIdWithItems_shouldNotCauseNPlusOne() {
//         // setup: save order with items
//         // act: call findByCustomerIdWithItems
//         // assert: verify only 1 SQL query executed (use @Sql or statistics)
//     }
// }

// --- 15d. WireMock (Component test — stub external services) ---
// @SpringBootTest(webEnvironment = RANDOM_PORT)
// @AutoConfigureWireMock(port = 0)  // random port, injected as wiremock.server.port
// class OrderComponentTest {
//
//     @Autowired TestRestTemplate restTemplate;
//
//     @Test
//     void createOrder_shouldReserveInventory() {
//         // Stub inventory service
//         stubFor(post(urlEqualTo("/api/v1/inventory/reserve"))
//                 .withRequestBody(matchingJsonPath("$.orderId"))
//                 .willReturn(aResponse()
//                         .withStatus(200)
//                         .withHeader("Content-Type", "application/json")
//                         .withBody("{\"reservationId\":\"res-1\",\"success\":true}")));
//
//         var request = new CreateOrderRequest("cust-1", List.of(), LocalDate.now().plusDays(1));
//         var response = restTemplate.postForEntity("/api/v1/orders", request, OrderResponse.class);
//
//         assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);
//         verify(postRequestedFor(urlEqualTo("/api/v1/inventory/reserve")));
//     }
//
//     @Test
//     void createOrder_shouldFallback_whenInventoryServiceIsDown() {
//         stubFor(post(urlEqualTo("/api/v1/inventory/reserve"))
//                 .willReturn(serverError()));   // 500 → triggers fallback
//
//         var response = restTemplate.postForEntity("/api/v1/orders",
//                 new CreateOrderRequest("cust-1", List.of(), LocalDate.now().plusDays(1)),
//                 OrderResponse.class);
//         // Should still return a graceful response via fallback, not 500
//         assertThat(response.getStatusCode()).isNotEqualTo(HttpStatus.INTERNAL_SERVER_ERROR);
//     }
// }

// --- 15e. Embedded Kafka Test ---
// @SpringBootTest
// @EmbeddedKafka(partitions = 1, topics = {"orders.placed", "inventory.commands"})
// class OrderSagaIntegrationTest {
//
//     @Autowired OrderSagaOrchestrator orchestrator;
//     @Autowired KafkaTemplate<String, Object> kafkaTemplate;
//
//     @Autowired EmbeddedKafkaBroker embeddedKafka;
//
//     @Test
//     void startSaga_shouldSendInventoryReserveCommand() throws Exception {
//         orchestrator.startSaga("order-1", "cust-1", BigDecimal.valueOf(100));
//
//         // Consume from inventory.commands and verify the message
//         Map<String, Object> consumerProps = KafkaTestUtils.consumerProps("test-group", "true", embeddedKafka);
//         try (var consumer = new DefaultKafkaConsumerFactory<String, String>(consumerProps).createConsumer()) {
//             embeddedKafka.consumeFromAnEmbeddedTopic(consumer, "inventory.commands");
//             ConsumerRecord<String, String> record = KafkaTestUtils.getSingleRecord(consumer, "inventory.commands");
//             assertThat(record.key()).isEqualTo("order-1");
//         }
//     }
// }

// ============================================================================
// QUICK-REFERENCE: TOP INTERVIEW TRAPS & ANSWERS
// ============================================================================

/**
 * TRAP 1: "Does @Transactional work on private methods?"
 * A: NO. Spring AOP proxy only intercepts public method calls from outside the bean.
 *    Self-invocation (this.doSomething()) bypasses the proxy.
 *    Fix: Inject self-reference (bad code smell) or restructure into a separate bean.
 *
 * TRAP 2: "What happens if Circuit Breaker + Retry are both active?"
 * A: Order matters. If CB is OPEN, Retry should not retry (it will fail immediately).
 *    Correct decoration order: CircuitBreaker(Retry(Bulkhead(call))).
 *    In Resilience4j: annotations apply inner-to-outer in method execution order.
 *
 * TRAP 3: "Can Feign client handle async calls?"
 * A: Standard Feign is blocking. For async: use WebClient or wrap Feign call in CompletableFuture.
 *    Spring Cloud OpenFeign 4.x has experimental reactive support.
 *
 * TRAP 4: "How do you handle token expiry in service-to-service calls?"
 * A: Use client credentials OAuth2 flow with Spring Security OAuth2 Client.
 *    Configure token refresh automatically via ClientRegistrationRepository.
 *    Spring handles token expiry and refreshes transparently.
 *
 * TRAP 5: "What is the difference between @Component, @Service, @Repository?"
 * A: All three are @Component specializations — identical in Spring DI behavior.
 *    @Repository: activates PersistenceExceptionTranslationPostProcessor
 *      → Spring DataAccessException wrapping of JPA/JDBC exceptions.
 *    @Service: semantic marker — no extra behavior.
 *    Distinction matters for: exception translation (@Repository) and AOP pointcuts.
 *
 * TRAP 6: "How does Spring handle circular dependencies?"
 * A: In Spring 6 / Boot 3.x, circular dependencies with constructor injection
 *    throw BeanCurrentlyInCreationException — they must be FIXED (redesign needed).
 *    Setter injection can break circular deps but is a design smell.
 *    If you see this, it indicates a design problem: introduce a mediator/event.
 *
 * TRAP 7: "What happens to Kafka offset if consumer throws an exception?"
 * A: With MANUAL_IMMEDIATE ack mode: offset is NOT committed → message is redelivered.
 *    With auto-commit: offset may already be committed before exception (data loss risk).
 *    DefaultErrorHandler retries with backoff, then sends to DLT after exhaustion.
 *
 * TRAP 8: "How do you prevent duplicate processing in Kafka consumers?"
 * A: Idempotency key: store processed event IDs in a DB table.
 *    Before processing, check if eventId already exists → skip if so.
 *    This makes consumers idempotent — safe to retry any number of times.
 *
 * TRAP 9: "What is the problem with EAGER loading on @OneToMany?"
 * A: EAGER on a collection uses an IN clause or extra SELECTs.
 *    If the parent has multiple EAGER collections, Hibernate may use a Cartesian product JOIN
 *    → MultipleBagFetchException or massive result sets.
 *    Rule: ALWAYS use LAZY on @OneToMany and @ManyToMany. Use JOIN FETCH only when needed.
 *
 * TRAP 10: "REST vs gRPC for inter-service communication?"
 * A: REST: human-readable (JSON), universal, easy tooling, HTTP/1.1 (request-response).
 *    gRPC: binary (Protobuf, 5-10x smaller), strongly typed, HTTP/2 (streaming, multiplexing),
 *          faster serialization, contract-first, code generation.
 *    Use gRPC for: high-throughput internal service calls, bi-directional streaming.
 *    Use REST for: external APIs, browser clients, simple integrations.
 *
 * TRAP 11: "How does Spring Cloud Gateway differ from a traditional reverse proxy (nginx)?"
 * A: Gateway is programmable — routes, filters, auth logic in Java/YAML.
 *    Integrates with service discovery, circuit breakers, rate limiting natively.
 *    nginx/Envoy: better raw performance, less JVM overhead, language-agnostic.
 *    In microservices: Spring Cloud Gateway for Spring ecosystem teams;
 *    nginx/Envoy/Kong as sidecar/infra-level in polyglot environments.
 *
 * TRAP 12: "What is eventual consistency and how do you handle it in the UI?"
 * A: After a write, the read model may not yet reflect the change (propagation delay).
 *    UI strategies: optimistic UI updates (assume success, rollback on failure),
 *    polling with exponential backoff, server-sent events / WebSocket for real-time push,
 *    showing "processing" state until confirmation arrives.
 */

