if(NOT BUILD_ANDROID_CURL)
    message("-- Using prebuilt curl library for Android")

    set(CURL_BIN_DIR ${PROJECT_SOURCE_DIR}/platform/android/prebuilt/curl)
    set(CURL_INCLUDES ${CURL_BIN_DIR}/include)

    set(CURL_STATIC_LIB ${CURL_BIN_DIR}/${ANDROID_ABI}/libcurl.a)
    set(SSL_STATIC_LIB ${CURL_BIN_DIR}/${ANDROID_ABI}/libssl.a)
    set(CRYPTO_STATIC_LIB ${CURL_BIN_DIR}/${ANDROID_ABI}/libcrypto.a)

    add_library(
        android-curl STATIC IMPORTED
    )
    set_target_properties(android-curl PROPERTIES IMPORTED_LOCATION ${CURL_STATIC_LIB})
    set_target_properties(android-curl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CURL_INCLUDES})
    set_target_properties(android-curl PROPERTIES INTERFACE_COMPILE_DEFINITIONS CURL_STATICLIB)
    set_target_properties(android-curl PROPERTIES INTERFACE_LINK_LIBRARIES "-l${SSL_STATIC_LIB} -l${CRYPTO_STATIC_LIB}")

    return()
else()
    message("-- Building curl for Android from source")
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/vendor/openssl OR NOT EXISTS ${PROJECT_SOURCE_DIR}/vendor/curl)
        message(FATAL_ERROR "Curl or OpenSSL sources missing - clone them to ${PROJECT_SOURCE_DIR}/vendor")
    endif()
endif()

include(ExternalProject)

include(ProcessorCount)
processorcount(NPROC)

# Check if Android NDK path is defined in the environment
if(DEFINED ENV{ANDROID_HOME})
    set(ANDROID_NDK_HOME $ENV{ANDROID_HOME}/ndk-bundle)
elseif(DEFINED ENV{ANDROID_SDK_ROOT})
    set(ANDROID_NDK_HOME $ENV{ANDROID_SDK_ROOT}/ndk-bundle)
elseif(DEFINED ANDROID_HOME)
    set(ANDROID_NDK_HOME ${ANDROID_HOME}/ndk-bundle)
elseif(DEFINED ANDROID_SDK_ROOT)
    set(ANDROID_NDK_HOME ${ANDROID_SDK_ROOT}/ndk-bundle)
else()
    message(FATAL_ERROR "Must set either ANDROID_HOME or ANDROID_SDK_ROOT variable")
endif()

if(CMAKE_HOST_SYSTEM_NAME STREQUAL Linux)
    set(HOST_TAG linux-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL Darwin)
    set(HOST_TAG darwin-x86_64)
else()
    message(FATAL_ERROR "Unsupported host: " ${CMAKE_SYSTEM_NAME})
endif()

if(${ANDROID_ABI} STREQUAL arm64-v8a)
    set(TRIPLE aarch64-linux-android)
    set(SSL_TARGET_ARCH android-arm64)
elseif(${ANDROID_ABI} STREQUAL x86_64)
    set(TRIPLE x86_64-linux-android)
    set(SSL_TARGET_ARCH android-x86_64)
else()
    message(FATAL_ERROR "-- Unsupported Android ABI: " ${ANDROID_ABI})
endif()

set(TOOLCHAIN ${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/${HOST_TAG})
set(PATH ${TOOLCHAIN}/bin:$ENV{PATH})

set(CC ${TOOLCHAIN}/bin/${TRIPLE}${ANDROID_NATIVE_API_LEVEL}-clang)
set(CXX ${TOOLCHAIN}/bin/${TRIPLE}${ANDROID_NATIVE_API_LEVEL}-clang++)
set(AR ${TOOLCHAIN}/bin/${TRIPLE}-ar)
set(RANLIB ${TOOLCHAIN}/bin/${TRIPLE}-ranlib)

# Build OpenSSL
set(SSL_SRC_DIR ${PROJECT_SOURCE_DIR}/vendor/openssl)
set(SSL_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SSL_INCLUDES ${SSL_BIN_DIR}/include)
file(MAKE_DIRECTORY ${SSL_INCLUDES})

set(SSL_STATIC_LIB ${SSL_BIN_DIR}/lib/libssl.a)
set(CRYPTO_STATIC_LIB ${SSL_BIN_DIR}/lib/libcrypto.a)

ExternalProject_Add(
    libssl
    PREFIX ${SSL_BIN_DIR}
    SOURCE_DIR ${SSL_SRC_DIR}
    DOWNLOAD_COMMAND
        cd
        ${SSL_SRC_DIR}
        &&
        echo
        "Starting openssl $<CONFIG>-${ANDROID_ABI} build"
    CONFIGURE_COMMAND
        echo
        "Configuring openssl $<CONFIG>-${ANDROID_ABI}"
        &&
        ${CMAKE_COMMAND}
        -E
        env
        ANDROID_NDK_HOME=${ANDROID_NDK_HOME}
        PATH=${PATH}
        ${SSL_SRC_DIR}/Configure
        ${SSL_TARGET_ARCH}
        no-asm
        no-shared
        no-tests
        no-unit-test
        zlib
        -static
        --prefix=${SSL_BIN_DIR}
    BUILD_COMMAND
        echo
        "Building openssl $<CONFIG>-${ANDROID_ABI}"
        &&
        ${CMAKE_COMMAND}
        -E
        env
        ANDROID_NDK_HOME=${ANDROID_NDK_HOME}
        PATH=${PATH}
        make
        -j${NPROC}
    INSTALL_COMMAND
        echo
        "Installing openssl $<CONFIG>-${ANDROID_ABI}"
        &&
        make
        install
    BUILD_BYPRODUCTS ${SSL_STATIC_LIB} ${CRYPTO_STATIC_LIB}
)

add_library(
    android-ssl STATIC IMPORTED
)
add_dependencies(android-ssl libssl)
set_target_properties(android-ssl PROPERTIES IMPORTED_LOCATION ${SSL_STATIC_LIB})
set_target_properties(android-ssl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SSL_INCLUDES})

add_library(
    android-crypto STATIC IMPORTED
)
add_dependencies(android-crypto libssl)
set_target_properties(android-crypto PROPERTIES IMPORTED_LOCATION ${CRYPTO_STATIC_LIB})
set_target_properties(android-crypto PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SSL_INCLUDES})

# Build CURL
set(CURL_DIR ${PROJECT_SOURCE_DIR}/vendor/curl)
set(CURL_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CURL_INCLUDES ${CURL_BIN_DIR}/include)
file(MAKE_DIRECTORY ${CURL_INCLUDES})
set(CURL_STATIC_LIB ${CURL_BIN_DIR}/lib/libcurl.a)

ExternalProject_Add(
    libcurl
    PREFIX ${CURL_BIN_DIR}
    DEPENDS libssl
    SOURCE_DIR ${CURL_DIR}
    DOWNLOAD_COMMAND
        cd
        ${CURL_DIR}
        &&
        echo
        "Starting curl $<CONFIG>-${ANDROID_ABI} build"
        &&
        ${CURL_DIR}/buildconf
    CONFIGURE_COMMAND
        echo
        "Configuring curl $<CONFIG>-${ANDROID_ABI}"
        &&
        CC=${CC}
        CXX=${CXX}
        AR=${AR}
        RANLIB=${RANLIB}
        ${CURL_DIR}/configure
        --host=${TRIPLE}
        --target=${TRIPLE}
        --with-sysroot=${TOOLCHAIN}/sysroot
        --with-ssl=${SSL_BIN_DIR}
        --enable-static
        --disable-shared
        --disable-verbose
        --enable-threaded-resolver
        --enable-libgcc
        --enable-ipv6
        --disable-manual
        --with-pic
        --prefix=${CURL_BIN_DIR}
    BUILD_COMMAND
        echo
        "Building curl $<CONFIG>-${ANDROID_ABI}"
        &&
        make
        -j${NPROC}
    INSTALL_COMMAND
        echo
        "Installing curl $<CONFIG>-${ANDROID_ABI}"
        &&
        make
        install
    BUILD_BYPRODUCTS ${CURL_STATIC_LIB}
)

add_library(
    android-curl STATIC IMPORTED
)
add_dependencies(libcurl libssl)
add_dependencies(
    android-curl
    libcurl
    android-ssl
    android-crypto
)
set_target_properties(android-curl PROPERTIES IMPORTED_LOCATION ${CURL_STATIC_LIB})
set_target_properties(android-curl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CURL_INCLUDES})
set_target_properties(android-curl PROPERTIES INTERFACE_COMPILE_DEFINITIONS CURL_STATICLIB)
set_target_properties(android-curl PROPERTIES INTERFACE_LINK_LIBRARIES "-l${SSL_STATIC_LIB} -l${CRYPTO_STATIC_LIB}")
