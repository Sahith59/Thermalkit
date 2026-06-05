import Foundation
import IOKit.hid

@_silgen_name("IOHIDServiceClientCopyEvent")
func _copyEvent(_ service: IOHIDServiceClient, _ type: Int32, _ options: Int32, _ timeout: Int32) -> AnyObject?
@_silgen_name("IOHIDEventGetFloatValue")
func _getFloatValue(_ event: AnyObject, _ field: Int32) -> Double

let client = IOHIDEventSystemClientCreateSimpleClient(kCFAllocatorDefault)
let services = (IOHIDEventSystemClientCopyServices(client) as? [IOHIDServiceClient]) ?? []
print("Total services: \(services.count)")

var thermalCount = 0
for service in services {
    let conforms = IOHIDServiceClientConformsTo(service, 0xFF00, 5)
    let name = (IOHIDServiceClientCopyProperty(service, "Product" as CFString) as? String) ?? "?"
    if conforms != 0 { thermalCount += 1 }
    if name.contains("PMU") || name.contains("tdie") || name.contains("tdev") || name.contains("gas") {
        let hasEvent = _copyEvent(service, 15, 0, 0) != nil
        print("  conforms=\(conforms) name=\(name) hasEvent=\(hasEvent)")
    }
}
print("Thermal-conforming services: \(thermalCount)")
