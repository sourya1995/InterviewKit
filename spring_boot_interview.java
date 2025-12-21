// ============================================================================
// SPRING BOOT INTERVIEW PREPARATION - COMPREHENSIVE GUIDE
// For 8+ Years SDE Experience
// ============================================================================

package com.interview.springboot;

import org.springframework.beans.factory.annotation.*;
import org.springframework.boot.*;
import org.springframework.boot.autoconfigure.*;
import org.springframework.context.annotation.*;
import org.springframework.stereotype.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.data.jpa.repository.*;
import org.springframework.transaction.annotation.*;
import org.springframework.cache.annotation.*;
import org.springframework.scheduling.annotation.*;
import org.springframework.security.config.annotation.web.builders.*;
import org.springframework.security.crypto.bcrypt.*;
import org.springframework.kafka.annotation.*;
import org.springframework.amqp.rabbit.annotation.*;
import org.springframework.retry.annotation.*;
import org.springframework.validation.annotation.*;
import org.springframework.web.client.*;
import org.springframework.http.*;
import org.springframework.data.domain.*;
import org.springframework.data.redis.core.*;
import org.springframework.cloud.client.circuitbreaker.*;
import org.springframework.boot.actuate.health.*;
import org.springframework.boot.context.properties.*;

import javax.persistence.*;
import javax.validation.constraints.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

// ============================================================================
// SPRING BOOT FUNDAMENTALS
// ============================================================================

/**
 * 1. Spring Boot Application Entry Point
 * - @SpringBootApplication combines @Configuration, @EnableAutoConfiguration, @ComponentScan
 * - Auto-configuration based on classpath and beans
 *
 * WHEN TO USE:
 * - Always, as the starting point of any Spring Boot application.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If auto-configuration causes issues, exclude specific classes using @SpringBootApplication(exclude = ...).
 * - If component scan is too slow, specify base packages explicitly.
 */
@SpringBootApplication
public class SpringBootInterviewApplication {
    
    public static void main(String[] args) {
        // Standard way to start Spring Boot
        SpringApplication.run(SpringBootInterviewApplication.class, args);
        
        // Customized startup
        SpringApplication app = new SpringApplication(SpringBootInterviewApplication.class);
        app.setBannerMode(Banner.Mode.OFF);
        app.setAdditionalProfiles("dev");
        app.run(args);
    }
    
    // Application lifecycle hooks
    @Bean
    public CommandLineRunner commandLineRunner() {
        return args -> {
            System.out.println("Application started with args: " + Arrays.toString(args));
        };
    }
    
    @Bean
    public ApplicationRunner applicationRunner() {
        return args -> {
            System.out.println("Application ready!");
        };
    }
}

/**
 * 2. Configuration Properties
 * - Type-safe configuration
 * - @ConfigurationProperties vs @Value
 *
 * WHEN TO USE:
 * - For grouping related properties (e.g., database, mail).
 * - When validation of properties is needed.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - Use @Value("${property}") for single, isolated values if creating a class is too much overhead.
 * - For legacy apps, use Environment.getProperty() but avoid this in new code.
 */
@Configuration
@ConfigurationProperties(prefix = "app")
@Validated
public class AppProperties {
    
    @NotNull
    private String name;
    
    @Min(1)
    @Max(65535)
    private int port;
    
    private Security security = new Security();
    private List<String> allowedOrigins = new ArrayList<>();
    private Map<String, String> customHeaders = new HashMap<>();
    
    // Nested configuration
    public static class Security {
        private boolean enabled = true;
        private String tokenSecret;
        private Duration tokenExpiration = Duration.ofHours(24);
        
        // Getters and setters
    }
    
    // Usage in application.yml:
    // app:
    //   name: MyApp
    //   port: 8080
    //   security:
    //     enabled: true
    //     token-secret: secret123
    //     token-expiration: 24h
    //   allowed-origins:
    //     - http://localhost:3000
    //   custom-headers:
    //     X-Custom: value
    
    // Getters and setters omitted for brevity
}

/**
 * 3. Dependency Injection Patterns
 *
 * WHEN TO USE:
 * - Constructor Injection: Mandatory dependencies (Best Practice).
 * - Setter Injection: Optional dependencies.
 * - Field Injection: Avoid if possible.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - Field injection (@Autowired) is acceptable for tests or very simple prototypes.
 * - If circular dependencies occur, use @Lazy or Setter injection to break the cycle.
 */
@Service
public class DependencyInjectionExamples {
    
    // Constructor injection (RECOMMENDED - immutable, testable)
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    
    public DependencyInjectionExamples(
            UserRepository userRepository,
            PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }
    
    // Field injection (NOT RECOMMENDED - hard to test, mutable)
    // @Autowired
    // private UserRepository userRepository;
    
    // Setter injection (use when optional dependency)
    private EmailService emailService;
    
    @Autowired(required = false)
    public void setEmailService(EmailService emailService) {
        this.emailService = emailService;
    }
    
    // Conditional injection
    @Autowired
    public DependencyInjectionExamples(
            UserRepository userRepository,
            PasswordEncoder passwordEncoder,
            @Qualifier("primaryCache") CacheManager cacheManager) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }
}

/**
 * 4. Bean Scopes and Lifecycle
 *
 * WHEN TO USE:
 * - Singleton: Default, stateless beans (Service, Repository).
 * - Prototype: Stateful beans, new instance per request.
 * - Request/Session: Web-specific state.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If Singleton beans hold state, ensure thread safety using synchronized blocks or Atomic variables.
 * - Avoid Prototype beans if injection cost is high; consider ObjectFactory.
 */
@Configuration
public class BeanScopeConfiguration {
    
    // Singleton (default) - one instance per Spring container
    @Bean
    @Scope("singleton")
    public SingletonBean singletonBean() {
        return new SingletonBean();
    }
    
    // Prototype - new instance every time
    @Bean
    @Scope("prototype")
    public PrototypeBean prototypeBean() {
        return new PrototypeBean();
    }
    
    // Request scope - one per HTTP request (web apps)
    @Bean
    @Scope(value = WebApplicationContext.SCOPE_REQUEST, proxyMode = ScopedProxyMode.TARGET_CLASS)
    public RequestScopedBean requestScopedBean() {
        return new RequestScopedBean();
    }
    
    // Session scope - one per HTTP session
    @Bean
    @Scope(value = WebApplicationContext.SCOPE_SESSION, proxyMode = ScopedProxyMode.TARGET_CLASS)
    public SessionScopedBean sessionScopedBean() {
        return new SessionScopedBean();
    }
    
    // Bean lifecycle callbacks
    @Bean(initMethod = "init", destroyMethod = "cleanup")
    public LifecycleBean lifecycleBean() {
        return new LifecycleBean();
    }
}

class LifecycleBean implements InitializingBean, DisposableBean {
    
    @PostConstruct
    public void postConstruct() {
        System.out.println("1. @PostConstruct called");
    }
    
    @Override
    public void afterPropertiesSet() {
        System.out.println("2. afterPropertiesSet called");
    }
    
    public void init() {
        System.out.println("3. Custom init method called");
    }
    
    @PreDestroy
    public void preDestroy() {
        System.out.println("4. @PreDestroy called");
    }
    
    @Override
    public void destroy() {
        System.out.println("5. destroy called");
    }
    
    public void cleanup() {
        System.out.println("6. Custom cleanup method called");
    }
}

// ============================================================================
// REST API DEVELOPMENT
// ============================================================================

/**
 * 5. REST Controller Best Practices
 *
 * WHEN TO USE:
 * - Exposing APIs to clients.
 * - Always use DTOs, never Entities.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If creating DTOs is too time-consuming for internal tools, returning Entities is acceptable but risky (lazy loading issues, exposing schema).
 * - Use @JsonIgnore on Entity fields to prevent data leakage if DTOs are skipped.
 */
@RestController
@RequestMapping("/api/v1/users")
@Validated
public class UserController {
    
    private final UserService userService;
    
    public UserController(UserService userService) {
        this.userService = userService;
    }
    
    // GET with path variable
    @GetMapping("/{id}")
    public ResponseEntity<UserDto> getUser(@PathVariable Long id) {
        return userService.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    // GET with query parameters and pagination
    @GetMapping
    public ResponseEntity<Page<UserDto>> getUsers(
            @RequestParam(required = false) String search,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "id,desc") String[] sort) {
        
        Pageable pageable = PageRequest.of(page, size, Sort.by(parseSort(sort)));
        Page<UserDto> users = userService.findAll(search, pageable);
        return ResponseEntity.ok(users);
    }
    
    // POST with validation
    @PostMapping
    public ResponseEntity<UserDto> createUser(
            @Valid @RequestBody CreateUserRequest request) {
        UserDto created = userService.create(request);
        URI location = ServletUriComponentsBuilder
                .fromCurrentRequest()
                .path("/{id}")
                .buildAndExpand(created.getId())
                .toUri();
        return ResponseEntity.created(location).body(created);
    }
    
    // PUT - full update
    @PutMapping("/{id}")
    public ResponseEntity<UserDto> updateUser(
            @PathVariable Long id,
            @Valid @RequestBody UpdateUserRequest request) {
        return ResponseEntity.ok(userService.update(id, request));
    }
    
    // PATCH - partial update
    @PatchMapping("/{id}")
    public ResponseEntity<UserDto> partialUpdateUser(
            @PathVariable Long id,
            @RequestBody Map<String, Object> updates) {
        return ResponseEntity.ok(userService.partialUpdate(id, updates));
    }
    
    // DELETE
    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteUser(@PathVariable Long id) {
        userService.delete(id);
    }
    
    // Custom headers and status
    @GetMapping("/{id}/profile")
    public ResponseEntity<byte[]> getUserProfile(@PathVariable Long id) {
        byte[] image = userService.getProfileImage(id);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_JPEG);
        headers.setCacheControl(CacheControl.maxAge(Duration.ofDays(7)));
        return new ResponseEntity<>(image, headers, HttpStatus.OK);
    }
    
    // Content negotiation
    @GetMapping(value = "/{id}", produces = {
            MediaType.APPLICATION_JSON_VALUE,
            MediaType.APPLICATION_XML_VALUE
    })
    public UserDto getUserWithContentNegotiation(@PathVariable Long id) {
        return userService.findById(id).orElseThrow();
    }
    
    private Sort.Order[] parseSort(String[] sort) {
        return Arrays.stream(sort)
                .map(s -> {
                    String[] parts = s.split(",");
                    return parts.length > 1 && parts[1].equalsIgnoreCase("desc")
                            ? Sort.Order.desc(parts[0])
                            : Sort.Order.asc(parts[0]);
                })
                .toArray(Sort.Order[]::new);
    }
}

/**
 * 6. Exception Handling
 *
 * WHEN TO USE:
 * - @RestControllerAdvice for global, consistent error responses.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If global handling is too generic, use local @ExceptionHandler in the Controller.
 * - For quick debugging, returning the stack trace (dev profile only) is acceptable, but never in production.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {
    
    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);
    
    // Handle custom business exceptions
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleResourceNotFound(
            ResourceNotFoundException ex,
            WebRequest request) {
        log.error("Resource not found: {}", ex.getMessage());
        ErrorResponse error = new ErrorResponse(
                HttpStatus.NOT_FOUND.value(),
                ex.getMessage(),
                request.getDescription(false),
                LocalDateTime.now()
        );
        return new ResponseEntity<>(error, HttpStatus.NOT_FOUND);
    }
    
    // Handle validation errors
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<Map<String, Object>> handleValidationErrors(
            MethodArgumentNotValidException ex) {
        Map<String, Object> errors = new HashMap<>();
        errors.put("timestamp", LocalDateTime.now());
        errors.put("status", HttpStatus.BAD_REQUEST.value());
        
        Map<String, String> fieldErrors = new HashMap<>();
        ex.getBindingResult().getFieldErrors().forEach(error -> 
            fieldErrors.put(error.getField(), error.getDefaultMessage())
        );
        errors.put("errors", fieldErrors);
        
        return new ResponseEntity<>(errors, HttpStatus.BAD_REQUEST);
    }
    
    // Handle constraint violations
    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ErrorResponse> handleConstraintViolation(
            ConstraintViolationException ex) {
        String message = ex.getConstraintViolations().stream()
                .map(ConstraintViolation::getMessage)
                .collect(Collectors.joining(", "));
        
        ErrorResponse error = new ErrorResponse(
                HttpStatus.BAD_REQUEST.value(),
                message,
                "",
                LocalDateTime.now()
        );
        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }
    
    // Handle access denied
    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ErrorResponse> handleAccessDenied(
            AccessDeniedException ex,
            WebRequest request) {
        ErrorResponse error = new ErrorResponse(
                HttpStatus.FORBIDDEN.value(),
                "Access denied",
                request.getDescription(false),
                LocalDateTime.now()
        );
        return new ResponseEntity<>(error, HttpStatus.FORBIDDEN);
    }
    
    // Handle all other exceptions
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGlobalException(
            Exception ex,
            WebRequest request) {
        log.error("Unexpected error", ex);
        ErrorResponse error = new ErrorResponse(
                HttpStatus.INTERNAL_SERVER_ERROR.value(),
                "An unexpected error occurred",
                request.getDescription(false),
                LocalDateTime.now()
        );
        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }
}

@Getter
@AllArgsConstructor
class ErrorResponse {
    private int status;
    private String message;
    private String path;
    private LocalDateTime timestamp;
}

// ============================================================================
// DATA ACCESS WITH JPA/HIBERNATE
// ============================================================================

/**
 * 7. Entity Design and Relationships
 *
 * WHEN TO USE:
 * - Mapping database tables to Java objects.
 * - Use JPA annotations for ORM.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If JPA relationships cause N+1 or performance issues, break the relationship and query by ID manually.
 * - Use @JsonIgnore to prevent infinite recursion in bidirectional relationships.
 */
@Entity
@Table(name = "users", indexes = {
        @Index(name = "idx_email", columnList = "email"),
        @Index(name = "idx_username", columnList = "username")
})
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User extends BaseEntity {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false, unique = true, length = 100)
    private String username;
    
    @Column(nullable = false, unique = true, length = 255)
    private String email;
    
    @Column(nullable = false)
    private String password;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private UserStatus status;
    
    @Column(name = "last_login")
    private LocalDateTime lastLogin;
    
    // One-to-Many relationship
    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    @Builder.Default
    private List<Order> orders = new ArrayList<>();
    
    // Many-to-Many relationship
    @ManyToMany(fetch = FetchType.LAZY)
    @JoinTable(
            name = "user_roles",
            joinColumns = @JoinColumn(name = "user_id"),
            inverseJoinColumns = @JoinColumn(name = "role_id")
    )
    @Builder.Default
    private Set<Role> roles = new HashSet<>();
    
    // One-to-One relationship
    @OneToOne(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true)
    private UserProfile profile;
    
    // Embedded object
    @Embedded
    private Address address;
    
    // JSON column (PostgreSQL)
    @Type(type = "json")
    @Column(columnDefinition = "json")
    private Map<String, Object> metadata;
    
    // Soft delete
    @Column(name = "deleted_at")
    private LocalDateTime deletedAt;
    
    // Version for optimistic locking
    @Version
    private Long version;
    
    // Helper methods for bidirectional relationships
    public void addOrder(Order order) {
        orders.add(order);
        order.setUser(this);
    }
    
    public void removeOrder(Order order) {
        orders.remove(order);
        order.setUser(null);
    }
    
    // Lifecycle callbacks
    @PrePersist
    protected void onCreate() {
        if (status == null) {
            status = UserStatus.ACTIVE;
        }
    }
    
    @PreUpdate
    protected void onUpdate() {
        // Validation or auto-update logic
    }
}

@MappedSuperclass
@EntityListeners(AuditingEntityListener.class)
@Getter
@Setter
abstract class BaseEntity {
    
    @CreatedDate
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
    
    @CreatedBy
    @Column(name = "created_by", updatable = false)
    private String createdBy;
    
    @LastModifiedBy
    @Column(name = "updated_by")
    private String updatedBy;
}

@Embeddable
@Getter
@Setter
class Address {
    private String street;
    private String city;
    private String state;
    private String zipCode;
    private String country;
}

/**
 * 8. Repository Patterns
 *
 * WHEN TO USE:
 * - Data access layer.
 * - Use JpaRepository for standard CRUD.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If JPA is too slow or complex for a query, use @Query(nativeQuery = true) or JdbcTemplate.
 * - For complex dynamic queries, use Criteria API or QueryDSL.
 */
// Basic repository
public interface UserRepository extends JpaRepository<User, Long> {
    
    // Query methods (derived from method name)
    Optional<User> findByEmail(String email);
    Optional<User> findByUsername(String username);
    List<User> findByStatus(UserStatus status);
    List<User> findByCreatedAtBetween(LocalDateTime start, LocalDateTime end);
    
    // @Query with JPQL
    @Query("SELECT u FROM User u WHERE u.email = :email AND u.status = :status")
    Optional<User> findByEmailAndStatus(@Param("email") String email, 
                                       @Param("status") UserStatus status);
    
    // @Query with native SQL
    @Query(value = "SELECT * FROM users WHERE email = :email", nativeQuery = true)
    Optional<User> findByEmailNative(@Param("email") String email);
    
    // Projection (DTO)
    @Query("SELECT new com.example.dto.UserSummaryDto(u.id, u.username, u.email) " +
           "FROM User u WHERE u.status = :status")
    List<UserSummaryDto> findUserSummariesByStatus(@Param("status") UserStatus status);
    
    // Custom query with pagination
    @Query("SELECT u FROM User u WHERE " +
           "(:search IS NULL OR LOWER(u.username) LIKE LOWER(CONCAT('%', :search, '%')) " +
           "OR LOWER(u.email) LIKE LOWER(CONCAT('%', :search, '%')))")
    Page<User> searchUsers(@Param("search") String search, Pageable pageable);
    
    // Modifying query
    @Modifying
    @Query("UPDATE User u SET u.status = :status WHERE u.id = :id")
    int updateStatus(@Param("id") Long id, @Param("status") UserStatus status);
    
    // Bulk delete
    @Modifying
    @Query("DELETE FROM User u WHERE u.status = :status AND u.createdAt < :date")
    int deleteInactiveUsers(@Param("status") UserStatus status, 
                           @Param("date") LocalDateTime date);
    
    // Count query
    @Query("SELECT COUNT(u) FROM User u WHERE u.status = :status")
    long countByStatus(@Param("status") UserStatus status);
    
    // Exists query
    boolean existsByEmail(String email);
    
    // Stream for large datasets
    @Query("SELECT u FROM User u")
    Stream<User> streamAll();
}

// Custom repository implementation
public interface UserRepositoryCustom {
    List<User> findByComplexCriteria(UserSearchCriteria criteria);
}

@Repository
public class UserRepositoryCustomImpl implements UserRepositoryCustom {
    
    @PersistenceContext
    private EntityManager entityManager;
    
    @Override
    public List<User> findByComplexCriteria(UserSearchCriteria criteria) {
        CriteriaBuilder cb = entityManager.getCriteriaBuilder();
        CriteriaQuery<User> query = cb.createQuery(User.class);
        Root<User> user = query.from(User.class);
        
        List<Predicate> predicates = new ArrayList<>();
        
        if (criteria.getUsername() != null) {
            predicates.add(cb.like(cb.lower(user.get("username")), 
                    "%" + criteria.getUsername().toLowerCase() + "%"));
        }
        
        if (criteria.getEmail() != null) {
            predicates.add(cb.equal(user.get("email"), criteria.getEmail()));
        }
        
        if (criteria.getStatus() != null) {
            predicates.add(cb.equal(user.get("status"), criteria.getStatus()));
        }
        
        if (criteria.getCreatedAfter() != null) {
            predicates.add(cb.greaterThanOrEqualTo(
                    user.get("createdAt"), criteria.getCreatedAfter()));
        }
        
        query.where(predicates.toArray(new Predicate[0]));
        
        return entityManager.createQuery(query).getResultList();
    }
}

/**
 * 9. Transaction Management
 *
 * WHEN TO USE:
 * - @Transactional on Service methods to ensure data integrity.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If @Transactional causes long database locks, reduce the scope or use programmatic TransactionTemplate.
 * - Split long-running processes into smaller transactions.
 */
@Service
@Transactional(readOnly = true)
public class UserService {
    
    private final UserRepository userRepository;
    private final EmailService emailService;
    private final AuditService auditService;
    
    public UserService(UserRepository userRepository, 
                      EmailService emailService,
                      AuditService auditService) {
        this.userRepository = userRepository;
        this.emailService = emailService;
        this.auditService = auditService;
    }
    
    // Read-only transaction (from class level)
    public Optional<UserDto> findById(Long id) {
        return userRepository.findById(id).map(this::toDto);
    }
    
    // Write transaction
    @Transactional
    public UserDto create(CreateUserRequest request) {
        // Validation
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new DuplicateResourceException("Email already exists");
        }
        
        // Create entity
        User user = User.builder()
                .username(request.getUsername())
                .email(request.getEmail())
                .password(passwordEncoder.encode(request.getPassword()))
                .status(UserStatus.ACTIVE)
                .build();
        
        // Save
        user = userRepository.save(user);
        
        // Send email (within same transaction)
        emailService.sendWelcomeEmail(user.getEmail());
        
        // Audit log
        auditService.logUserCreation(user.getId());
        
        return toDto(user);
    }
    
    // Transaction with custom settings
    @Transactional(
            propagation = Propagation.REQUIRED,
            isolation = Isolation.READ_COMMITTED,
            timeout = 30,
            rollbackFor = Exception.class,
            noRollbackFor = ValidationException.class
    )
    public void complexOperation() {
        // Complex business logic
    }
    
    // Programmatic transaction management
    @Autowired
    private PlatformTransactionManager transactionManager;
    
    public void programmaticTransaction() {
        TransactionTemplate transactionTemplate = new TransactionTemplate(transactionManager);
        transactionTemplate.setTimeout(30);
        
        transactionTemplate.execute(status -> {
            try {
                // Transaction logic
                userRepository.save(new User());
                return null;
            } catch (Exception e) {
                status.setRollbackOnly();
                throw e;
            }
        });
    }
    
    // Handling N+1 problem with fetch join
    @Transactional(readOnly = true)
    public List<UserWithOrdersDto> getUsersWithOrders() {
        // Bad: N+1 problem
        // List<User> users = userRepository.findAll();
        // users.forEach(user -> user.getOrders().size());
        
        // Good: Fetch join
        return userRepository.findAllWithOrders()
                .stream()
                .map(this::toUserWithOrdersDto)
                .collect(Collectors.toList());
    }
    
    // Batch operations
    @Transactional
    public void batchInsert(List<CreateUserRequest> requests) {
        int batchSize = 50;
        for (int i = 0; i < requests.size(); i++) {
            User user = toEntity(requests.get(i));
            userRepository.save(user);
            
            if (i % batchSize == 0 && i > 0) {
                // Flush and clear to avoid memory issues
                userRepository.flush();
                entityManager.clear();
            }
        }
    }
    
    @PersistenceContext
    private EntityManager entityManager;
    
    private UserDto toDto(User user) {
        // Mapping logic
        return null;
    }
    
    private User toEntity(CreateUserRequest request) {
        // Mapping logic
        return null;
    }
}

// ============================================================================
// CACHING
// ============================================================================

/**
 * 10. Caching Strategies
 *
 * WHEN TO USE:
 * - To improve read performance for frequently accessed, rarely changed data.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If distributed cache (Redis) is too complex, use simple in-memory (Caffeine) but watch heap usage.
 * - If stale data is an issue, reduce TTL or implement manual eviction.
 */
@Configuration
@EnableCaching
public class CacheConfiguration {
    
    // Caffeine cache (in-memory)
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager("users", "products");
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .maximumSize(1000)
                .expireAfterWrite(Duration.ofMinutes(10))
                .recordStats());
        return cacheManager;
    }
    
    // Redis cache
    @Bean
    public RedisCacheManager redisCacheManager(RedisConnectionFactory connectionFactory) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig()
                .entryTtl(Duration.ofMinutes(10))
                .serializeKeysWith(
                        RedisSerializationContext.SerializationPair.fromSerializer(
                                new StringRedisSerializer()))
                .serializeValuesWith(
                        RedisSerializationContext.SerializationPair.fromSerializer(
                                new GenericJackson2JsonRedisSerializer()));
        
        return RedisCacheManager.builder(connectionFactory)
                .cacheDefaults(config)
                .build();
    }
    
    // Custom cache resolver
    @Bean
    public CacheResolver cacheResolver(CacheManager cacheManager) {
        return new SimpleCacheResolver(cacheManager);
    }
}

@Service
public class CachedUserService {
    
    private final UserRepository userRepository;
    
    // Cache result
    @Cacheable(value = "users", key = "#id")
    public Optional<UserDto> findById(Long id) {
        return userRepository.findById(id).map(this::toDto);
    }
    
    // Cache with condition
    @Cacheable(value = "users", key = "#email", condition = "#email != null")
    public Optional<UserDto> findByEmail(String email) {
        return userRepository.findByEmail(email).map(this::toDto);
    }
    
    // Cache with unless (don't cache if result is null)
    @Cacheable(value = "users", key = "#id", unless = "#result == null")
    public UserDto getUserById(Long id) {
        return userRepository.findById(id)
                .map(this::toDto)
                .orElse(null);
    }
    
    // Evict cache entry
    @CacheEvict(value = "users", key = "#id")
    public void deleteUser(Long id) {
        userRepository.deleteById(id);
    }
    
    // Evict all cache entries
    @CacheEvict(value = "users", allEntries = true)
    public void clearCache() {
        // Cache cleared
    }
    
    // Update cache
    @CachePut(value = "users", key = "#result.id")
    public UserDto updateUser(Long id, UpdateUserRequest request) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
        // Update logic
        return toDto(userRepository.save(user));
    }
    
    // Multiple cache operations
    @Caching(
            evict = {
                    @CacheEvict(value = "users", key = "#id"),
                    @CacheEvict(value = "userStats", key = "#id")
            }
    )
    public void complexDelete(Long id) {
        userRepository.deleteById(id);
    }
    
    // Custom cache key generator
    @Cacheable(value = "users", keyGenerator = "customKeyGenerator")
    public List<UserDto> findByComplexCriteria(UserSearchCriteria criteria) {
        return userRepository.findByComplexCriteria(criteria)
                .stream()
                .map(this::toDto)
                .collect(Collectors.toList());
    }
    
    private UserDto toDto(User user) {
        return null;
    }
}

@Component
public class CustomKeyGenerator implements KeyGenerator {
    
    @Override
    public Object generate(Object target, Method method, Object... params) {
        return method.getName() + "_" + 
               Arrays.stream(params)
                       .map(Object::toString)
                       .collect(Collectors.joining("_"));
    }
}

// ============================================================================
// SECURITY
// ============================================================================

/**
 * 11. Spring Security Configuration
 *
 * WHEN TO USE:
 * - Securing endpoints, handling authentication and authorization.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If full OAuth2/OIDC is too complex to set up initially, use Basic Auth over HTTPS for internal services.
 * - For legacy apps, you might need to disable CSRF, but ensure other protections are in place.
 */
@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true, securedEnabled = true)
public class SecurityConfiguration {
    
    private final JwtAuthenticationFilter jwtAuthFilter;
    private final UserDetailsService userDetailsService;
    
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .csrf().disable()
                .cors()
                .and()
                .authorizeHttpRequests()
                    .antMatchers("/api/v1/auth/**").permitAll()
                    .antMatchers("/actuator/health").permitAll()
                    .antMatchers("/swagger-ui/**", "/v3/api-docs/**").permitAll()
                    .antMatchers(HttpMethod.GET, "/api/v1/products/**").permitAll()
                    .antMatchers("/api/v1/admin/**").hasRole("ADMIN")
                    .anyRequest().authenticated()
                .and()
                .sessionManagement()
                    .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                .and()
                .authenticationProvider(authenticationProvider())
                .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class)
                .exceptionHandling()
                    .authenticationEntryPoint(
                            (request, response, authException) -> {
                                response.setStatus(HttpStatus.UNAUTHORIZED.value());
                                response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                                response.getWriter().write(
                                        "{\"error\": \"Unauthorized\", \"message\": \"" + 
                                        authException.getMessage() + "\"}");
                            })
                    .accessDeniedHandler(
                            (request, response, accessDeniedException) -> {
                                response.setStatus(HttpStatus.FORBIDDEN.value());
                                response.setContentType(MediaType.APPLICATION_JSON_VALUE);
                                response.getWriter().write(
                                        "{\"error\": \"Forbidden\", \"message\": \"" + 
                                        accessDeniedException.getMessage() + "\"}");
                            });
        
        return http.build();
    }
    
    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
        authProvider.setUserDetailsService(userDetailsService);
        authProvider.setPasswordEncoder(passwordEncoder());
        return authProvider;
    }
    
    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) 
            throws Exception {
        return config.getAuthenticationManager();
    }
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder(12);
    }
    
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList("http://localhost:3000"));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        configuration.setMaxAge(3600L);
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }
}

/**
 * 12. JWT Authentication Implementation
 *
 * WHEN TO USE:
 * - Stateless authentication for REST APIs.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If implementing full token rotation is too hard, use short-lived access tokens and force re-login.
 * - Don't store sensitive data in JWT claims.
 */
@Component
public class JwtService {
    
    @Value("${jwt.secret}")
    private String secretKey;
    
    @Value("${jwt.expiration}")
    private long jwtExpiration;
    
    @Value("${jwt.refresh-expiration}")
    private long refreshExpiration;
    
    public String generateToken(UserDetails userDetails) {
        Map<String, Object> claims = new HashMap<>();
        return generateToken(claims, userDetails);
    }
    
    public String generateToken(Map<String, Object> extraClaims, UserDetails userDetails) {
        return buildToken(extraClaims, userDetails, jwtExpiration);
    }
    
    public String generateRefreshToken(UserDetails userDetails) {
        return buildToken(new HashMap<>(), userDetails, refreshExpiration);
    }
    
    private String buildToken(
            Map<String, Object> extraClaims,
            UserDetails userDetails,
            long expiration) {
        return Jwts.builder()
                .setClaims(extraClaims)
                .setSubject(userDetails.getUsername())
                .setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + expiration))
                .signWith(getSignInKey(), SignatureAlgorithm.HS256)
                .compact();
    }
    
    public boolean isTokenValid(String token, UserDetails userDetails) {
        final String username = extractUsername(token);
        return (username.equals(userDetails.getUsername())) && !isTokenExpired(token);
    }
    
    private boolean isTokenExpired(String token) {
        return extractExpiration(token).before(new Date());
    }
    
    private Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }
    
    public String extractUsername(String token) {
        return extractClaim(token, Claims::getSubject);
    }
    
    public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }
    
    private Claims extractAllClaims(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(getSignInKey())
                .build()
                .parseClaimsJws(token)
                .getBody();
    }
    
    private Key getSignInKey() {
        byte[] keyBytes = Decoders.BASE64.decode(secretKey);
        return Keys.hmacShaKeyFor(keyBytes);
    }
}

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    
    private final JwtService jwtService;
    private final UserDetailsService userDetailsService;
    
    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain) throws ServletException, IOException {
        
        final String authHeader = request.getHeader("Authorization");
        
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            filterChain.doFilter(request, response);
            return;
        }
        
        try {
            final String jwt = authHeader.substring(7);
            final String userEmail = jwtService.extractUsername(jwt);
            
            if (userEmail != null && SecurityContextHolder.getContext().getAuthentication() == null) {
                UserDetails userDetails = userDetailsService.loadUserByUsername(userEmail);
                
                if (jwtService.isTokenValid(jwt, userDetails)) {
                    UsernamePasswordAuthenticationToken authToken = 
                            new UsernamePasswordAuthenticationToken(
                                    userDetails,
                                    null,
                                    userDetails.getAuthorities());
                    
                    authToken.setDetails(
                            new WebAuthenticationDetailsSource().buildDetails(request));
                    
                    SecurityContextHolder.getContext().setAuthentication(authToken);
                }
            }
        } catch (Exception e) {
            logger.error("Cannot set user authentication", e);
        }
        
        filterChain.doFilter(request, response);
    }
}

/**
 * 13. Method-level Security
 *
 * WHEN TO USE:
 * - Fine-grained authorization at the service layer.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If @PreAuthorize expressions get too complex, move the logic to a dedicated SecurityService and call it.
 * - If performance is hit, do bulk checks instead of per-item checks.
 */
@Service
public class SecuredService {
    
    // Only users with ADMIN role can access
    @PreAuthorize("hasRole('ADMIN')")
    public void adminOnlyMethod() {
        // Admin logic
    }
    
    // Multiple roles
    @PreAuthorize("hasAnyRole('ADMIN', 'MANAGER')")
    public void managerMethod() {
        // Manager logic
    }
    
    // SpEL expression
    @PreAuthorize("hasRole('USER') and #userId == authentication.principal.id")
    public void updateOwnProfile(Long userId, UpdateProfileRequest request) {
        // User can only update their own profile
    }
    
    // Check ownership
    @PreAuthorize("@securityService.isOwner(#orderId, authentication.principal.id)")
    public void updateOrder(Long orderId, UpdateOrderRequest request) {
        // Logic
    }
    
    // Post-authorization (after method execution)
    @PostAuthorize("returnObject.userId == authentication.principal.id")
    public Order getOrder(Long orderId) {
        return orderRepository.findById(orderId).orElseThrow();
    }
    
    // Filtering collections
    @PostFilter("filterObject.userId == authentication.principal.id")
    public List<Order> getUserOrders() {
        return orderRepository.findAll();
    }
    
    @PreFilter("filterObject.userId == authentication.principal.id")
    public void deleteOrders(List<Long> orderIds) {
        orderRepository.deleteAllById(orderIds);
    }
}

@Service
public class SecurityService {
    
    private final OrderRepository orderRepository;
    
    public boolean isOwner(Long orderId, Long userId) {
        return orderRepository.findById(orderId)
                .map(order -> order.getUserId().equals(userId))
                .orElse(false);
    }
}

// ============================================================================
// ASYNC PROCESSING
// ============================================================================

/**
 * 14. Async Methods and Thread Pools
 *
 * WHEN TO USE:
 * - Fire-and-forget tasks (emails, logging).
 * - Parallel processing.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If debugging async code is difficult, use a flag to run synchronously in development.
 * - Always configure a custom TaskExecutor; default one is not production-ready.
 */
@Configuration
@EnableAsync
public class AsyncConfiguration implements AsyncConfigurer {
    
    @Override
    @Bean(name = "taskExecutor")
    public Executor getAsyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(10);
        executor.setQueueCapacity(25);
        executor.setThreadNamePrefix("async-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.setAwaitTerminationSeconds(60);
        executor.initialize();
        return executor;
    }
    
    @Override
    public AsyncUncaughtExceptionHandler getAsyncUncaughtExceptionHandler() {
        return (ex, method, params) -> {
            System.err.println("Exception in async method: " + method.getName());
            ex.printStackTrace();
        };
    }
    
    // Custom executor for specific use cases
    @Bean(name = "mailExecutor")
    public Executor mailExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(2);
        executor.setMaxPoolSize(5);
        executor.setQueueCapacity(10);
        executor.setThreadNamePrefix("mail-");
        executor.initialize();
        return executor;
    }
}

@Service
public class AsyncService {
    
    // Simple async method
    @Async
    public void asyncMethod() {
        System.out.println("Executing async method in thread: " + 
                Thread.currentThread().getName());
    }
    
    // Async method with return value
    @Async
    public CompletableFuture<String> asyncMethodWithReturn() {
        try {
            Thread.sleep(1000);
            return CompletableFuture.completedFuture("Result");
        } catch (InterruptedException e) {
            return CompletableFuture.failedFuture(e);
        }
    }
    
    // Using specific executor
    @Async("mailExecutor")
    public void sendEmailAsync(String to, String subject, String body) {
        // Send email logic
    }
    
    // Combining multiple async operations
    public CompletableFuture<OrderSummary> getOrderSummary(Long orderId) {
        CompletableFuture<Order> orderFuture = getOrderAsync(orderId);
        CompletableFuture<User> userFuture = getUserAsync(orderId);
        CompletableFuture<List<OrderItem>> itemsFuture = getOrderItemsAsync(orderId);
        
        return CompletableFuture.allOf(orderFuture, userFuture, itemsFuture)
                .thenApply(v -> {
                    Order order = orderFuture.join();
                    User user = userFuture.join();
                    List<OrderItem> items = itemsFuture.join();
                    return new OrderSummary(order, user, items);
                });
    }
    
    @Async
    public CompletableFuture<Order> getOrderAsync(Long orderId) {
        return CompletableFuture.completedFuture(orderRepository.findById(orderId).orElse(null));
    }
}

// ============================================================================
// SCHEDULING
// ============================================================================

/**
 * 15. Scheduled Tasks
 *
 * WHEN TO USE:
 * - Periodic background jobs.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If running multiple instances, use ShedLock to prevent duplicate execution.
 * - If Cron expressions are confusing, use fixedRate for simple intervals.
 */
@Configuration
@EnableScheduling
public class SchedulingConfiguration implements SchedulingConfigurer {
    
    @Override
    public void configureTasks(ScheduledTaskRegistrar taskRegistrar) {
        taskRegistrar.setScheduler(taskScheduler());
    }
    
    @Bean
    public TaskScheduler taskScheduler() {
        ThreadPoolTaskScheduler scheduler = new ThreadPoolTaskScheduler();
        scheduler.setPoolSize(5);
        scheduler.setThreadNamePrefix("scheduled-");
        scheduler.setWaitForTasksToCompleteOnShutdown(true);
        scheduler.setAwaitTerminationSeconds(60);
        scheduler.initialize();
        return scheduler;
    }
}

@Component
public class ScheduledTasks {
    
    private static final Logger log = LoggerFactory.getLogger(ScheduledTasks.class);
    
    // Fixed rate - executes every 5 seconds (regardless of previous execution)
    @Scheduled(fixedRate = 5000)
    public void fixedRateTask() {
        log.info("Fixed rate task - {}", LocalDateTime.now());
    }
    
    // Fixed delay - waits 5 seconds after previous execution completes
    @Scheduled(fixedDelay = 5000)
    public void fixedDelayTask() {
        log.info("Fixed delay task - {}", LocalDateTime.now());
    }
    
    // Initial delay
    @Scheduled(initialDelay = 10000, fixedRate = 5000)
    public void taskWithInitialDelay() {
        log.info("Task with initial delay - {}", LocalDateTime.now());
    }
    
    // Cron expression
    @Scheduled(cron = "0 0 * * * *") // Every hour
    public void cronTask() {
        log.info("Cron task - {}", LocalDateTime.now());
    }
    
    // Cron with timezone
    @Scheduled(cron = "0 0 9 * * *", zone = "America/New_York")
    public void cronTaskWithTimezone() {
        log.info("Cron task with timezone - {}", LocalDateTime.now());
    }
    
    // Conditional scheduling
    @Scheduled(cron = "${app.cleanup.cron:0 0 2 * * *}")
    @ConditionalOnProperty(name = "app.cleanup.enabled", havingValue = "true")
    public void conditionalTask() {
        log.info("Conditional cleanup task - {}", LocalDateTime.now());
    }
    
    // Lock-based scheduling (prevent concurrent execution in clustered environment)
    @Scheduled(cron = "0 */5 * * * *")
    @SchedulerLock(name = "processDataTask", 
                   lockAtMostFor = "4m", 
                   lockAtLeastFor = "1m")
    public void processDataTask() {
        log.info("Processing data - {}", LocalDateTime.now());
        // Long-running task
    }
}

// ============================================================================
// MESSAGING - KAFKA
// ============================================================================

/**
 * 16. Kafka Integration
 *
 * WHEN TO USE:
 * - High-throughput, distributed event streaming.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If Kafka is overkill (too much infra), use RabbitMQ or Redis Pub/Sub.
 * - If message ordering is critical but hard to guarantee, use a single partition or key-based ordering.
 */
@Configuration
public class KafkaConfiguration {
    
    @Value("${spring.kafka.bootstrap-servers}")
    private String bootstrapServers;
    
    // Producer configuration
    @Bean
    public ProducerFactory<String, Object> producerFactory() {
        Map<String, Object> config = new HashMap<>();
        config.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        config.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        config.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, JsonSerializer.class);
        config.put(ProducerConfig.ACKS_CONFIG, "all");
        config.put(ProducerConfig.RETRIES_CONFIG, 3);
        config.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        return new DefaultKafkaProducerFactory<>(config);
    }
    
    @Bean
    public KafkaTemplate<String, Object> kafkaTemplate() {
        return new KafkaTemplate<>(producerFactory());
    }
    
    // Consumer configuration
    @Bean
    public ConsumerFactory<String, Object> consumerFactory() {
        Map<String, Object> config = new HashMap<>();
        config.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        config.put(ConsumerConfig.GROUP_ID_CONFIG, "user-service-group");
        config.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        config.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        config.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        config.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);
        config.put(JsonDeserializer.TRUSTED_PACKAGES, "*");
        return new DefaultKafkaConsumerFactory<>(config);
    }
    
    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, Object> kafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, Object> factory = 
                new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory());
        factory.setConcurrency(3);
        factory.getContainerProperties().setAckMode(ContainerProperties.AckMode.MANUAL);
        factory.setCommonErrorHandler(new DefaultErrorHandler(
                new FixedBackOff(1000L, 3L)));
        return factory;
    }
}

@Service
public class KafkaProducerService {
    
    private static final Logger log = LoggerFactory.getLogger(KafkaProducerService.class);
    private final KafkaTemplate<String, Object> kafkaTemplate;
    
    // Simple send
    public void sendMessage(String topic, Object message) {
        kafkaTemplate.send(topic, message);
    }
    
    // Send with key
    public void sendMessageWithKey(String topic, String key, Object message) {
        kafkaTemplate.send(topic, key, message);
    }
    
    // Send with callback
    public void sendWithCallback(String topic, Object message) {
        ListenableFuture<SendResult<String, Object>> future = 
                kafkaTemplate.send(topic, message);
        
        future.addCallback(
                result -> log.info("Message sent successfully: {}", result.getRecordMetadata()),
                ex -> log.error("Failed to send message", ex)
        );
    }
    
    // Transactional send
    @Transactional("kafkaTransactionManager")
    public void sendTransactional(String topic, Object message) {
        kafkaTemplate.send(topic, message);
        kafkaTemplate.send(topic + "-audit", new AuditEvent(message));
    }
}

@Component
public class KafkaConsumerService {
    
    private static final Logger log = LoggerFactory.getLogger(KafkaConsumerService.class);
    
    // Simple consumer
    @KafkaListener(topics = "user-events", groupId = "user-service-group")
    public void consume(UserEvent event) {
        log.info("Consumed event: {}", event);
        // Process event
    }
    
    // Consumer with manual acknowledgment
    @KafkaListener(topics = "order-events", groupId = "order-service-group")
    public void consumeWithAck(
            @Payload OrderEvent event,
            @Header(KafkaHeaders.RECEIVED_PARTITION_ID) int partition,
            @Header(KafkaHeaders.OFFSET) long offset,
            Acknowledgment acknowledgment) {
        
        try {
            log.info("Consumed from partition {}, offset {}: {}", partition, offset, event);
            // Process event
            acknowledgment.acknowledge();
        } catch (Exception e) {
            log.error("Error processing event", e);
            // Don't acknowledge - will retry
        }
    }
    
    // Batch consumer
    @KafkaListener(topics = "batch-events", groupId = "batch-group")
    public void consumeBatch(List<UserEvent> events) {
        log.info("Consumed batch of {} events", events.size());
        events.forEach(this::processEvent);
    }
    
    // Multiple topics
    @KafkaListener(topics = {"topic1", "topic2"}, groupId = "multi-topic-group")
    public void consumeMultipleTopics(ConsumerRecord<String, Object> record) {
        log.info("Consumed from topic {}: {}", record.topic(), record.value());
    }
    
    // Topic pattern
    @KafkaListener(topicPattern = "user-.*", groupId = "pattern-group")
    public void consumePattern(Object message) {
        log.info("Consumed: {}", message);
    }
    
    private void processEvent(UserEvent event) {
        // Processing logic
    }
}

// ============================================================================
// MESSAGING - RABBITMQ
// ============================================================================

/**
 * 17. RabbitMQ Integration
 *
 * WHEN TO USE:
 * - Traditional message queuing, complex routing.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If exchange/queue binding is complex, use the default exchange for simple point-to-point.
 * - Ensure DLQ (Dead Letter Queue) is configured to avoid losing messages.
 */
@Configuration
public class RabbitMQConfiguration {
    
    public static final String QUEUE_NAME = "user-queue";
    public static final String EXCHANGE_NAME = "user-exchange";
    public static final String ROUTING_KEY = "user.created";
    
    @Bean
    public Queue queue() {
        return QueueBuilder.durable(QUEUE_NAME)
                .withArgument("x-message-ttl", 60000) // TTL 60 seconds
                .withArgument("x-max-length", 1000) // Max 1000 messages
                .build();
    }
    
    @Bean
    public TopicExchange exchange() {
        return new TopicExchange(EXCHANGE_NAME);
    }
    
    @Bean
    public Binding binding(Queue queue, TopicExchange exchange) {
        return BindingBuilder
                .bind(queue)
                .to(exchange)
                .with(ROUTING_KEY);
    }
    
    // Dead letter queue
    @Bean
    public Queue deadLetterQueue() {
        return QueueBuilder.durable("user-queue-dlq").build();
    }
    
    @Bean
    public TopicExchange deadLetterExchange() {
        return new TopicExchange("user-exchange-dlq");
    }
    
    @Bean
    public Binding deadLetterBinding(Queue deadLetterQueue, TopicExchange deadLetterExchange) {
        return BindingBuilder
                .bind(deadLetterQueue)
                .to(deadLetterExchange)
                .with("user.#");
    }
    
    @Bean
    public MessageConverter messageConverter() {
        return new Jackson2JsonMessageConverter();
    }
    
    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        template.setMessageConverter(messageConverter());
        return template;
    }
}

@Service
public class RabbitMQProducerService {
    
    private final RabbitTemplate rabbitTemplate;
    
    public void sendMessage(Object message) {
        rabbitTemplate.convertAndSend(
                RabbitMQConfiguration.EXCHANGE_NAME,
                RabbitMQConfiguration.ROUTING_KEY,
                message);
    }
    
    public void sendWithProperties(Object message) {
        rabbitTemplate.convertAndSend(
                RabbitMQConfiguration.EXCHANGE_NAME,
                RabbitMQConfiguration.ROUTING_KEY,
                message,
                msg -> {
                    msg.getMessageProperties().setExpiration("60000");
                    msg.getMessageProperties().setPriority(5);
                    msg.getMessageProperties().setHeader("custom-header", "value");
                    return msg;
                });
    }
}

@Component
public class RabbitMQConsumerService {
    
    private static final Logger log = LoggerFactory.getLogger(RabbitMQConsumerService.class);
    
    @RabbitListener(queues = RabbitMQConfiguration.QUEUE_NAME)
    public void receiveMessage(UserEvent event) {
        log.info("Received message: {}", event);
        // Process message
    }
    
    @RabbitListener(queues = RabbitMQConfiguration.QUEUE_NAME, 
                    ackMode = "MANUAL")
    public void receiveWithManualAck(
            UserEvent event,
            Channel channel,
            @Header(AmqpHeaders.DELIVERY_TAG) long tag) {
        try {
            log.info("Received: {}", event);
            // Process message
            channel.basicAck(tag, false);
        } catch (Exception e) {
            try {
                channel.basicNack(tag, false, true); // Requeue
            } catch (IOException ex) {
                log.error("Error nacking message", ex);
            }
        }
    }
}

// ============================================================================
// RESILIENCE - CIRCUIT BREAKER, RETRY, RATE LIMITING
// ============================================================================

/**
 * 18. Resilience4j - Circuit Breaker
 *
 * WHEN TO USE:
 * - Preventing cascading failures in microservices.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If Circuit Breaker adds too much latency/complexity, start with simple Timeouts.
 * - Use a fallback method that returns a default value or cached data.
 */
@Service
public class ResilientService {
    
    private static final Logger log = LoggerFactory.getLogger(ResilientService.class);
    private final RestTemplate restTemplate;
    
    // Circuit breaker
    @CircuitBreaker(name = "userService", fallbackMethod = "getUserFallback")
    public User getUser(Long id) {
        log.info("Calling external user service");
        return restTemplate.getForObject("http://user-service/users/" + id, User.class);
    }
    
    public User getUserFallback(Long id, Exception ex) {
        log.error("Circuit breaker fallback for user {}", id, ex);
        return User.builder()
                .id(id)
                .username("fallback-user")
                .build();
    }
    
    // Retry mechanism
    @Retry(name = "userService", fallbackMethod = "getUserFallback")
    public User getUserWithRetry(Long id) {
        log.info("Attempting to call user service (will retry on failure)");
        return restTemplate.getForObject("http://user-service/users/" + id, User.class);
    }
    
    // Rate limiter
    @RateLimiter(name = "userService")
    public User getUserWithRateLimit(Long id) {
        return restTemplate.getForObject("http://user-service/users/" + id, User.class);
    }
    
    // Bulkhead (limit concurrent calls)
    @Bulkhead(name = "userService", fallbackMethod = "getUserFallback")
    public User getUserWithBulkhead(Long id) {
        return restTemplate.getForObject("http://user-service/users/" + id, User.class);
    }
    
    // Time limiter
    @TimeLimiter(name = "userService", fallbackMethod = "getUserFallback")
    public CompletableFuture<User> getUserWithTimeout(Long id) {
        return CompletableFuture.supplyAsync(() -> 
            restTemplate.getForObject("http://user-service/users/" + id, User.class));
    }
    
    // Combined resilience patterns
    @CircuitBreaker(name = "orderService")
    @Retry(name = "orderService")
    @RateLimiter(name = "orderService")
    public Order getOrder(Long id) {
        return restTemplate.getForObject("http://order-service/orders/" + id, Order.class);
    }
}

// Configuration in application.yml:
// resilience4j:
//   circuitbreaker:
//     instances:
//       userService:
//         slidingWindowSize: 10
//         minimumNumberOfCalls: 5
//         failureRateThreshold: 50
//         waitDurationInOpenState: 10s
//         permittedNumberOfCallsInHalfOpenState: 3
//   retry:
//     instances:
//       userService:
//         maxAttempts: 3
//         waitDuration: 1s
//         retryExceptions:
//           - java.net.SocketTimeoutException
//   ratelimiter:
//     instances:
//       userService:
//         limitForPeriod: 10
//         limitRefreshPeriod: 1s
//         timeoutDuration: 0s

/**
 * 19. Spring Retry
 *
 * WHEN TO USE:
 * - Transient failures (network glitches).
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - Ensure operations are Idempotent before retrying.
 * - If retry storms occur, use exponential backoff with jitter.
 */
@Service
@EnableRetry
public class RetryService {
    
    // Basic retry
    @Retryable(
            value = {RestClientException.class},
            maxAttempts = 3,
            backoff = @Backoff(delay = 1000))
    public String callExternalService() {
        // Call that might fail
        return restTemplate.getForObject("http://api.example.com/data", String.class);
    }
    
    @Recover
    public String recoverFromFailure(RestClientException ex) {
        log.error("All retry attempts failed", ex);
        return "fallback-data";
    }
    
    // Exponential backoff
    @Retryable(
            value = {IOException.class},
            maxAttempts = 5,
            backoff = @Backoff(delay = 1000, multiplier = 2, maxDelay = 10000))
    public void uploadFile(byte[] data) {
        // Upload logic
    }
}

// ============================================================================
// REST CLIENT INTEGRATION
// ============================================================================

/**
 * 20. RestTemplate and WebClient
 *
 * WHEN TO USE:
 * - Calling external REST APIs.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - RestTemplate is in maintenance mode; use WebClient for new development.
 * - If WebClient is too complex (reactive), RestTemplate is still supported and fine for blocking scenarios.
 */
@Configuration
public class RestClientConfiguration {
    
    // RestTemplate (synchronous)
    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(Duration.ofSeconds(5))
                .setReadTimeout(Duration.ofSeconds(30))
                .interceptors(new LoggingInterceptor())
                .errorHandler(new CustomResponseErrorHandler())
                .build();
    }
    
    // WebClient (async, reactive)
    @Bean
    public WebClient webClient(WebClient.Builder builder) {
        return builder
                .baseUrl("https://api.example.com")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .defaultHeader(HttpHeaders.USER_AGENT, "MyApp/1.0")
                .filter(ExchangeFilterFunction.ofRequestProcessor(clientRequest -> {
                    log.info("Request: {} {}", clientRequest.method(), clientRequest.url());
                    return Mono.just(clientRequest);
                }))
                .filter(ExchangeFilterFunction.ofResponseProcessor(clientResponse -> {
                    log.info("Response status: {}", clientResponse.statusCode());
                    return Mono.just(clientResponse);
                }))
                .build();
    }
}

@Service
public class RestClientService {
    
    private final RestTemplate restTemplate;
    private final WebClient webClient;
    
    // RestTemplate GET
    public User getUserRestTemplate(Long id) {
        String url = "http://user-service/users/{id}";
        return restTemplate.getForObject(url, User.class, id);
    }
    
    // RestTemplate POST
    public User createUserRestTemplate(CreateUserRequest request) {
        String url = "http://user-service/users";
        ResponseEntity<User> response = restTemplate.postForEntity(url, request, User.class);
        return response.getBody();
    }
    
    // RestTemplate with headers
    public User getUserWithHeaders(Long id) {
        String url = "http://user-service/users/{id}";
        HttpHeaders headers = new HttpHeaders();
        headers.set("Authorization", "Bearer token");
        HttpEntity<Void> entity = new HttpEntity<>(headers);
        
        ResponseEntity<User> response = restTemplate.exchange(
                url, HttpMethod.GET, entity, User.class, id);
        return response.getBody();
    }
    
    // WebClient GET (async)
    public Mono<User> getUserWebClient(Long id) {
        return webClient.get()
                .uri("/users/{id}", id)
                .retrieve()
                .onStatus(HttpStatus::is4xxClientError, response -> 
                    Mono.error(new ResourceNotFoundException("User not found")))
                .onStatus(HttpStatus::is5xxServerError, response -> 
                    Mono.error(new ServiceException("Service unavailable")))
                .bodyToMono(User.class);
    }
    
    // WebClient POST
    public Mono<User> createUserWebClient(CreateUserRequest request) {
        return webClient.post()
                .uri("/users")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(User.class);
    }
    
    // WebClient with timeout
    public Mono<User> getUserWithTimeout(Long id) {
        return webClient.get()
                .uri("/users/{id}", id)
                .retrieve()
                .bodyToMono(User.class)
                .timeout(Duration.ofSeconds(5))
                .onErrorResume(TimeoutException.class, e -> 
                    Mono.error(new ServiceException("Request timed out")));
    }
    
    // Parallel requests with WebClient
    public Mono<UserWithOrders> getUserWithOrders(Long userId) {
        Mono<User> userMono = webClient.get()
                .uri("/users/{id}", userId)
                .retrieve()
                .bodyToMono(User.class);
        
        Mono<List<Order>> ordersMono = webClient.get()
                .uri("/users/{id}/orders", userId)
                .retrieve()
                .bodyToFlux(Order.class)
                .collectList();
        
        return Mono.zip(userMono, ordersMono, UserWithOrders::new);
    }
}

// ============================================================================
// VALIDATION
// ============================================================================

/**
 * 21. Input Validation
 *
 * WHEN TO USE:
 * - Validating incoming DTOs before processing.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If Bean Validation (@Valid) is too rigid, perform manual validation checks in the Service layer.
 * - Return consistent error structures even for manual validation errors.
 */
@Getter
@Setter
public class CreateUserRequest {
    
    @NotBlank(message = "Username is required")
    @Size(min = 3, max = 50, message = "Username must be between 3 and 50 characters")
    @Pattern(regexp = "^[a-zA-Z0-9_]+$", message = "Username can only contain letters, numbers, and underscores")
    private String username;
    
    @NotBlank(message = "Email is required")
    @Email(message = "Email must be valid")
    private String email;
    
    @NotBlank(message = "Password is required")
    @Size(min = 8, message = "Password must be at least 8 characters")
    @Pattern(regexp = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).*$", 
             message = "Password must contain at least one lowercase, uppercase, and digit")
    private String password;
    
    @NotNull(message = "Age is required")
    @Min(value = 18, message = "Must be at least 18 years old")
    @Max(value = 120, message = "Age must be less than 120")
    private Integer age;
    
    @NotNull(message = "Terms acceptance is required")
    @AssertTrue(message = "You must accept the terms and conditions")
    private Boolean acceptedTerms;
    
    @Valid
    @NotNull(message = "Address is required")
    private AddressDto address;
    
    @NotEmpty(message = "At least one phone number is required")
    private List<@Pattern(regexp = "^\\+?[1-9]\\d{1,14}$", message = "Invalid phone number") String> phoneNumbers;
    
    @PastOrPresent(message = "Birth date cannot be in the future")
    private LocalDate birthDate;
    
    @FutureOrPresent(message = "Start date must be in the present or future")
    private LocalDate startDate;
    
    // Custom validation
    @ValidPassword
    private String customPassword;
}

// Custom validator
@Target({ElementType.FIELD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = PasswordValidator.class)
public @interface ValidPassword {
    String message() default "Invalid password";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
}

public class PasswordValidator implements ConstraintValidator<ValidPassword, String> {
    
    @Override
    public boolean isValid(String password, ConstraintValidatorContext context) {
        if (password == null) return false;
        
        // Custom validation logic
        return password.length() >= 8 
                && password.matches(".*[A-Z].*") 
                && password.matches(".*[a-z].*")
                && password.matches(".*\\d.*")
                && password.matches(".*[@#$%^&+=].*");
    }
}

// Class-level validation
@Constraint(validatedBy = PasswordMatchesValidator.class)
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface PasswordMatches {
    String message() default "Passwords don't match";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
}

public class PasswordMatchesValidator implements ConstraintValidator<PasswordMatches, Object> {
    
    @Override
    public boolean isValid(Object obj, ConstraintValidatorContext context) {
        if (obj instanceof CreateUserRequest) {
            CreateUserRequest request = (CreateUserRequest) obj;
            // Compare password and confirmPassword fields
            return true; // Simplified
        }
        return false;
    }
}

// Validation groups
public interface BasicInfo {}
public interface AdvancedInfo {}

@Getter
@Setter
public class UserValidationGroups {
    
    @NotBlank(groups = BasicInfo.class)
    private String username;
    
    @NotBlank(groups = {BasicInfo.class, AdvancedInfo.class})
    @Email
    private String email;
    
    @NotBlank(groups = AdvancedInfo.class)
    @Size(min = 10, max = 1000)
    private String bio;
}

@RestController
public class ValidationController {
    
    // Validate specific group
    @PostMapping("/users/basic")
    public ResponseEntity<?> createBasicUser(
            @Validated(BasicInfo.class) @RequestBody UserValidationGroups user) {
        return ResponseEntity.ok(user);
    }
    
    @PostMapping("/users/advanced")
    public ResponseEntity<?> createAdvancedUser(
            @Validated({BasicInfo.class, AdvancedInfo.class}) @RequestBody UserValidationGroups user) {
        return ResponseEntity.ok(user);
    }
    
    // Programmatic validation
    @Autowired
    private Validator validator;
    
    public void manualValidation(CreateUserRequest request) {
        Set<ConstraintViolation<CreateUserRequest>> violations = validator.validate(request);
        
        if (!violations.isEmpty()) {
            String errors = violations.stream()
                    .map(ConstraintViolation::getMessage)
                    .collect(Collectors.joining(", "));
            throw new ValidationException(errors);
        }
    }
}

// ============================================================================
// MONITORING AND ACTUATOR
// ============================================================================

/**
 * 22. Spring Boot Actuator
 *
 * WHEN TO USE:
 * - Monitoring and managing the application in production.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If security is a concern and you can't configure Spring Security, expose Actuator on a different, internal-only port (management.server.port).
 * - Disable sensitive endpoints (heapdump, env) if not strictly needed.
 */
@Configuration
public class ActuatorConfiguration {
    
    // Custom health indicator
    @Component
    public class DatabaseHealthIndicator implements HealthIndicator {
        
        private final DataSource dataSource;
        
        @Override
        public Health health() {
            try (Connection connection = dataSource.getConnection()) {
                if (connection.isValid(1)) {
                    return Health.up()
                            .withDetail("database", "PostgreSQL")
                            .withDetail("connection", "Active")
                            .build();
                }
            } catch (Exception e) {
                return Health.down()
                        .withDetail("error", e.getMessage())
                        .build();
            }
            return Health.down().build();
        }
    }
    
    // Custom info contributor
    @Component
    public class CustomInfoContributor implements InfoContributor {
        
        @Override
        public void contribute(Info.Builder builder) {
            builder.withDetail("app", Map.of(
                    "name", "MyApp",
                    "version", "1.0.0",
                    "description", "Sample Spring Boot Application"
            ));
            builder.withDetail("team", Map.of(
                    "name", "Development Team",
                    "email", "dev@example.com"
            ));
        }
    }
    
    // Custom metrics
    @Component
    public class CustomMetrics {
        
        private final MeterRegistry meterRegistry;
        private final Counter userCreationCounter;
        private final Timer apiResponseTimer;
        
        public CustomMetrics(MeterRegistry meterRegistry) {
            this.meterRegistry = meterRegistry;
            
            this.userCreationCounter = Counter.builder("users.created")
                    .description("Total number of users created")
                    .tag("service", "user-service")
                    .register(meterRegistry);
            
            this.apiResponseTimer = Timer.builder("api.response.time")
                    .description("API response time")
                    .register(meterRegistry);
        }
        
        public void incrementUserCreation() {
            userCreationCounter.increment();
        }
        
        public void recordApiResponseTime(Runnable operation) {
            apiResponseTimer.record(operation);
        }
        
        // Gauge example (for current values)
        public void registerGauge(String name, Supplier<Number> supplier) {
            Gauge.builder(name, supplier)
                    .register(meterRegistry);
        }
    }
    
    // Custom endpoint
    @Component
    @Endpoint(id = "custom")
    public class CustomEndpoint {
        
        @ReadOperation
        public Map<String, Object> customEndpoint() {
            return Map.of(
                    "status", "OK",
                    "timestamp", LocalDateTime.now(),
                    "data", "Custom endpoint data"
            );
        }
        
        @WriteOperation
        public void updateSomething(@Selector String name, String value) {
            // Update logic
        }
        
        @DeleteOperation
        public void deleteSomething(@Selector String name) {
            // Delete logic
        }
    }
}

// Actuator endpoints available:
// GET  /actuator/health          - Health check
// GET  /actuator/info            - Application info
// GET  /actuator/metrics         - Metrics
// GET  /actuator/prometheus      - Prometheus metrics
// GET  /actuator/env             - Environment properties
// GET  /actuator/loggers         - Logger configuration
// POST /actuator/loggers/{name}  - Change log level
// GET  /actuator/httptrace       - HTTP trace
// GET  /actuator/threaddump      - Thread dump
// GET  /actuator/heapdump        - Heap dump

// ============================================================================
// FILE UPLOAD AND DOWNLOAD
// ============================================================================

/**
 * 23. File Operations
 *
 * WHEN TO USE:
 * - Handling file uploads and downloads.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If local storage fills up, switch to cloud storage (S3, Azure Blob) using a similar interface.
 * - If large file uploads fail, adjust max-file-size and use streaming.
 */
@RestController
@RequestMapping("/api/v1/files")
public class FileController {
    
    private final FileStorageService fileStorageService;
    
    // Single file upload
    @PostMapping("/upload")
    public ResponseEntity<FileUploadResponse> uploadFile(
            @RequestParam("file") MultipartFile file) {
        
        validateFile(file);
        String fileName = fileStorageService.storeFile(file);
        
        String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                .path("/api/v1/files/download/")
                .path(fileName)
                .toUriString();
        
        return ResponseEntity.ok(new FileUploadResponse(
                fileName,
                fileDownloadUri,
                file.getContentType(),
                file.getSize()
        ));
    }
    
    // Multiple files upload
    @PostMapping("/upload-multiple")
    public ResponseEntity<List<FileUploadResponse>> uploadMultipleFiles(
            @RequestParam("files") MultipartFile[] files) {
        
        List<FileUploadResponse> responses = Arrays.stream(files)
                .map(file -> {
                    validateFile(file);
                    String fileName = fileStorageService.storeFile(file);
                    String fileDownloadUri = ServletUriComponentsBuilder.fromCurrentContextPath()
                            .path("/api/v1/files/download/")
                            .path(fileName)
                            .toUriString();
                    return new FileUploadResponse(
                            fileName, fileDownloadUri, file.getContentType(), file.getSize());
                })
                .collect(Collectors.toList());
        
        return ResponseEntity.ok(responses);
    }
    
    // Download file
    @GetMapping("/download/{fileName:.+}")
    public ResponseEntity<Resource> downloadFile(
            @PathVariable String fileName,
            HttpServletRequest request) {
        
        Resource resource = fileStorageService.loadFileAsResource(fileName);
        
        String contentType = null;
        try {
            contentType = request.getServletContext().getMimeType(resource.getFile().getAbsolutePath());
        } catch (IOException ex) {
            contentType = "application/octet-stream";
        }
        
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType(contentType))
                .header(HttpHeaders.CONTENT_DISPOSITION, 
                        "attachment; filename=\"" + resource.getFilename() + "\"")
                .body(resource);
    }
    
    // Stream large file
    @GetMapping("/stream/{fileName:.+}")
    public ResponseEntity<StreamingResponseBody> streamFile(@PathVariable String fileName) {
        
        Resource resource = fileStorageService.loadFileAsResource(fileName);
        
        StreamingResponseBody stream = outputStream -> {
            try (InputStream inputStream = resource.getInputStream()) {
                byte[] buffer = new byte[1024];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            }
        };
        
        return ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .header(HttpHeaders.CONTENT_DISPOSITION, 
                        "attachment; filename=\"" + fileName + "\"")
                .body(stream);
    }
    
    // Delete file
    @DeleteMapping("/{fileName:.+}")
    public ResponseEntity<String> deleteFile(@PathVariable String fileName) {
        fileStorageService.deleteFile(fileName);
        return ResponseEntity.ok("File deleted successfully");
    }

    // List all files
    @GetMapping
    public ResponseEntity<List<String>> listFiles() {
        List<String> fileNames = fileStorageService.loadAll()
                .map(path -> path.getFileName().toString())
                .collect(Collectors.toList());
        return ResponseEntity.ok(fileNames);
    }

    private void validateFile(MultipartFile file) {
        if (file.isEmpty()) {
            throw new ValidationException("File is empty");
        }
        
        // Validate file size (e.g., max 10MB)
        long maxSize = 10 * 1024 * 1024;
        if (file.getSize() > maxSize) {
            throw new ValidationException("File size exceeds maximum limit of 10MB");
        }
        
        // Validate file type
        String contentType = file.getContentType();
        List<String> allowedTypes = Arrays.asList(
                "image/jpeg", "image/png", "application/pdf");
        
        if (contentType == null || !allowedTypes.contains(contentType)) {
            throw new ValidationException("File type not allowed");
        }
    }
}

@Service
public class FileStorageService {
    
    private final Path fileStorageLocation;
    
    @Autowired
    public FileStorageService(@Value("${file.upload-dir}") String uploadDir) {
        this.fileStorageLocation = Paths.get(uploadDir).toAbsolutePath().normalize();
        
        try {
            Files.createDirectories(this.fileStorageLocation);
        } catch (Exception ex) {
            throw new FileStorageException("Could not create upload directory", ex);
        }
    }
    
    public String storeFile(MultipartFile file) {
        String fileName = StringUtils.cleanPath(file.getOriginalFilename());
        
        try {
            // Check for invalid characters
            if (fileName.contains("..")) {
                throw new FileStorageException("Invalid file path: " + fileName);
            }
            
            // Add timestamp to avoid name conflicts
            String newFileName = System.currentTimeMillis() + "_" + fileName;
            Path targetLocation = this.fileStorageLocation.resolve(newFileName);
            Files.copy(file.getInputStream(), targetLocation, StandardCopyOption.REPLACE_EXISTING);
            
            return newFileName;
        } catch (IOException ex) {
            throw new FileStorageException("Could not store file " + fileName, ex);
        }
    }
    
    public Resource loadFileAsResource(String fileName) {
        try {
            Path filePath = this.fileStorageLocation.resolve(fileName).normalize();
            Resource resource = new UrlResource(filePath.toUri());
            
            if (resource.exists()) {
                return resource;
            } else {
                throw new ResourceNotFoundException("File not found: " + fileName);
            }
        } catch (MalformedURLException ex) {
            throw new ResourceNotFoundException("File not found: " + fileName, ex);
        }
    }
    
    public void deleteFile(String fileName) {
        try {
            Path filePath = this.fileStorageLocation.resolve(fileName).normalize();
            Files.deleteIfExists(filePath);
        } catch (IOException ex) {
            throw new FileStorageException("Could not delete file: " + fileName, ex);
        }
    }

    public Stream<Path> loadAll() {
        try {
            return Files.walk(this.fileStorageLocation, 1)
                    .filter(path -> !path.equals(this.fileStorageLocation))
                    .map(this.fileStorageLocation::relativize);
        } catch (IOException e) {
            throw new FileStorageException("Failed to read stored files", e);
        }
    }
}

// ============================================================================
// TESTING
// ============================================================================

/**
 * 24. Testing (JUnit 5, Mockito, Testcontainers)
 *
 * WHEN TO USE:
 * - Ensuring code correctness and preventing regressions.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If Integration Tests (@SpringBootTest) are too slow, rely more on Unit Tests with Mockito.
 * - Use Testcontainers for realistic DB tests instead of H2, if performance allows.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class ApplicationTests {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private UserService userService;

    @Test
    void contextLoads() {
        // Verify context loads
    }

    @Test
    void shouldReturnUser() throws Exception {
        UserDto user = new UserDto(1L, "testuser", "test@example.com");
        Mockito.when(userService.findById(1L)).thenReturn(Optional.of(user));

        mockMvc.perform(get("/api/v1/users/1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.username").value("testuser"));
    }
}

@DataJpaTest
class UserRepositoryTest {

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    private UserRepository userRepository;

    @Test
    void findByEmail_thenReturnUser() {
        // given
        User user = new User();
        user.setEmail("alex@example.com");
        user.setUsername("alex");
        user.setPassword("password");
        user.setStatus(UserStatus.ACTIVE);
        entityManager.persist(user);
        entityManager.flush();

        // when
        Optional<User> found = userRepository.findByEmail(user.getEmail());

        // then
        assertThat(found).isPresent();
        assertThat(found.get().getUsername()).isEqualTo(user.getUsername());
    }
}

// ============================================================================
// MICROSERVICES
// ============================================================================

/**
 * 25. Microservices Communication (Feign, Discovery)
 *
 * WHEN TO USE:
 * - Distributed systems with multiple services.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If Microservices complexity is too high, start with a Modular Monolith.
 * - If Feign is too "magic", use WebClient for more control.
 */
@EnableFeignClients
@Configuration
class MicroserviceConfig {
    // Feign client configuration
}

@FeignClient(name = "order-service", fallback = OrderServiceFallback.class)
interface OrderServiceClient {
    @GetMapping("/orders/{id}")
    OrderDto getOrder(@PathVariable("id") Long id);
}

@Component
class OrderServiceFallback implements OrderServiceClient {
    @Override
    public OrderDto getOrder(Long id) {
        return new OrderDto(); // Fallback
    }
}

// ============================================================================
// WEBSOCKETS
// ============================================================================

/**
 * 26. WebSockets with STOMP
 *
 * WHEN TO USE:
 * - Real-time, bi-directional communication (chat, notifications).
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If WebSocket scaling is difficult, use an external Message Broker (RabbitMQ/ActiveMQ) as the relay.
 * - For simple notifications, Server-Sent Events (SSE) might be easier.
 */
@Configuration
@EnableWebSocketMessageBroker
class WebSocketConfig implements WebSocketMessageBrokerConfigurer {

    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.enableSimpleBroker("/topic");
        config.setApplicationDestinationPrefixes("/app");
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws").withSockJS();
    }
}

@Controller
class ChatController {

    @MessageMapping("/chat")
    @SendTo("/topic/messages")
    public ChatMessage send(ChatMessage message) {
        return new ChatMessage(message.getContent(), LocalDateTime.now());
    }
}

@Getter
@AllArgsConstructor
class ChatMessage {
    private String content;
    private LocalDateTime timestamp;
}

// ============================================================================
// GRAPHQL
// ============================================================================

/**
 * 27. GraphQL
 *
 * WHEN TO USE:
 * - Flexible APIs where clients need specific data fields.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If N+1 problems persist, use DataLoaders or @BatchMapping.
 * - If GraphQL complexity is high, stick to REST for simple resources.
 */
@Controller
class UserGraphQLController {

    private final UserService userService;

    public UserGraphQLController(UserService userService) {
        this.userService = userService;
    }

    @QueryMapping
    public UserDto userById(@Argument Long id) {
        return userService.findById(id).orElse(null);
    }

    @SchemaMapping
    public List<Order> orders(UserDto user) {
        // Fetch orders for user (solving N+1 with @BatchMapping if needed)
        return new ArrayList<>(); 
    }
}

// ============================================================================
// REACTIVE PROGRAMMING
// ============================================================================

/**
 * 28. Reactive Programming (WebFlux)
 *
 * WHEN TO USE:
 * - High-concurrency, non-blocking applications.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - If debugging Reactive streams is too hard, stick to the Servlet stack (Blocking) unless you have massive scale requirements.
 * - Don't mix Blocking and Non-Blocking code; it kills performance.
 */
@RestController
@RequestMapping("/api/v1/reactive/users")
class ReactiveUserController {

    private final ReactiveUserRepository userRepository;

    public ReactiveUserController(ReactiveUserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @GetMapping("/{id}")
    public Mono<User> getUser(@PathVariable Long id) {
        return userRepository.findById(id);
    }

    @GetMapping
    public Flux<User> getAllUsers() {
        return userRepository.findAll();
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Mono<User> createUser(@RequestBody User user) {
        return userRepository.save(user);
    }
}

interface ReactiveUserRepository extends ReactiveCrudRepository<User, Long> {
    @Query("SELECT * FROM users WHERE email = :email")
    Mono<User> findByEmail(String email);
}

// ============================================================================
// DESIGN PATTERNS
// ============================================================================

/**
 * 29. Design Patterns in Spring
 *
 * WHEN TO USE:
 * - Solving common architectural problems cleanly.
 *
 * MITIGATION WHEN BEST PRACTICES FAIL:
 * - Don't over-engineer. If a simple if-else works, don't force a Strategy pattern.
 * - Use patterns to simplify code, not to show off.
 */
// Strategy Pattern
interface PaymentStrategy {
    void pay(double amount);
}

@Component("creditCard")
class CreditCardPayment implements PaymentStrategy {
    public void pay(double amount) { /* ... */ }
}

@Component("paypal")
class PayPalPayment implements PaymentStrategy {
    public void pay(double amount) { /* ... */ }
}

@Service
class PaymentService {
    private final Map<String, PaymentStrategy> strategies;

    public PaymentService(Map<String, PaymentStrategy> strategies) {
        this.strategies = strategies;
    }

    public void processPayment(String type, double amount) {
        strategies.get(type).pay(amount);
    }
}

// Observer Pattern (Events)
@Getter
@AllArgsConstructor
class UserCreatedEvent {
    private Long userId;
    private String email;
}

@Component
class UserEventListener {
    @EventListener
    public void handleUserCreated(UserCreatedEvent event) {
        // Send welcome email, etc.
        System.out.println("User created: " + event.getEmail());
    }
}
