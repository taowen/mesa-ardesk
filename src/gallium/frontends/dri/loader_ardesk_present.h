#ifndef LOADER_ARDESK_PRESENT_H
#define LOADER_ARDESK_PRESENT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool loader_ardesk_available(void);
int loader_ardesk_connect(void);
int loader_ardesk_alloc(int sock, uint32_t window, uint32_t w, uint32_t h,
                        uint32_t fourcc, uint32_t *token, int *fd,
                        uint32_t *pitch, uint32_t *offset, uint32_t *got_fourcc);
int loader_ardesk_present(int sock, uint32_t window, uint32_t w, uint32_t h,
                          uint32_t token);
int loader_ardesk_release(int sock, uint32_t token);

#ifdef __cplusplus
}
#endif

#endif
