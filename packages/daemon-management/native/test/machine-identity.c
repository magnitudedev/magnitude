#include "../machine-identity-data.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static const unsigned char system_record[] = {
  1,27,0,0,1,2,3,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,
  'L','e','n','o','v','o',0,'2','1','H','D',0,'T','1','4',0,'T','h','i','n','k','P','a','d',0,'S','E','C','R','E','T',0,0
};
static const unsigned char chassis_record[] = {3,6,0,0,0,0x8a,0,0};
int main(void) {
  unsigned char table[1024]; magnitude_machine_identity value = {0};
  memcpy(table, system_record, sizeof(system_record));
  memcpy(table + sizeof(system_record), chassis_record, sizeof(chassis_record));
  magnitude_parse_smbios(table, sizeof(system_record) + sizeof(chassis_record), &value);
  assert(!strcmp(value.manufacturer,"Lenovo") && !strcmp(value.model,"21HD"));
  assert(!strcmp(value.version,"T14") && !strcmp(value.family,"ThinkPad"));
  assert(value.chassis_type == 10); /* mask the lock bit */
  assert(!strstr(value.manufacturer,"SECRET") && !strstr(value.model,"SECRET"));
  memset(&value,0,sizeof(value));
  memcpy(table,chassis_record,sizeof(chassis_record));
  memcpy(table+sizeof(chassis_record),system_record,sizeof(system_record));
  magnitude_parse_smbios(table,sizeof(system_record)+sizeof(chassis_record),&value);
  assert(value.chassis_type == 10 && !strcmp(value.model,"21HD"));
  /* Every truncated prefix is safe, and only complete structures contribute. */
  for (size_t n=0;n<sizeof(system_record);n++) {
    memset(&value,0,sizeof(value)); magnitude_parse_smbios(system_record,n,&value);
    assert(!value.model[0] && !value.manufacturer[0]);
  }
  memcpy(table,system_record,sizeof(system_record));table[4]=99;table[5]=0;
  memset(&value,0,sizeof(value));magnitude_parse_smbios(table,sizeof(system_record),&value);
  assert(!value.model[0] && !value.manufacturer[0]);
  memset(table,'X',sizeof(table));table[0]=1;table[1]=6;table[4]=1;table[5]=1;table[306]=0;table[307]=0;
  memset(&value,0,sizeof(value));magnitude_parse_smbios(table,308,&value);
  assert(!value.model[0]); /* overlong strings are rejected, never truncated */
  table[0]=127;table[1]=4;table[4]=0;table[5]=0;
  memcpy(table+6,system_record,sizeof(system_record));
  memset(&value,0,sizeof(value));magnitude_parse_smbios(table,6+sizeof(system_record),&value);assert(!value.model[0]);
  /* Deterministic malformed tables exercise bounds under ASan and UBSan. */
  uint32_t random=1234567;
  for (unsigned round=0;round<10000;round++) {
    for (unsigned i=0;i<sizeof(table);i++) {random=random*1664525u+1013904223u;table[i]=(unsigned char)(random>>24);}
    memset(&value,0,sizeof(value));magnitude_parse_smbios(table,round%sizeof(table),&value);
  }
  puts("SMBIOS fixtures passed");return 0;
}
